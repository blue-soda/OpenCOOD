#!/usr/bin/env python
"""Benchmark the opt-in CoBEVFlow GPU ROI implementation.

The script accepts a copied config and the original checkpoint directory, so
the original config/model files and checkpoint are never modified.  It uses
the same fixed-sample, CUDA-synchronized protocol as the existing stage
profiler and records top-level module timings plus the complete
``generate_box_flow`` wall time.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils


def stats(values):
    return {
        'median_ms': statistics.median(values) if values else None,
        'p95_ms': float(np.percentile(values, 95)) if values else None,
        'mean_ms': statistics.mean(values) if values else None,
    }


class ModuleTimer:
    def __init__(self, model, names):
        self.calls = defaultdict(lambda: defaultdict(list))
        self.current_iter = -1
        self.pending = []
        self.handles = []
        for name in names:
            module = getattr(model, name, None)
            if not isinstance(module, torch.nn.Module):
                continue
            self.handles.append(module.register_forward_pre_hook(self._pre(name)))
            self.handles.append(module.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(module, inputs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.pending.append((name, start, end))
        return hook

    def _post(self, name):
        def hook(module, inputs, output):
            for index in range(len(self.pending) - 1, -1, -1):
                if self.pending[index][0] == name:
                    old_name, start, end = self.pending[index]
                    end.record()
                    self.pending[index] = (old_name, start, end)
                    break
        return hook

    def flush(self):
        torch.cuda.synchronize()
        current = self.calls[self.current_iter]
        for name, start, end in self.pending:
            current[name].append(float(start.elapsed_time(end)))
        self.pending = []

    def close(self):
        for handle in self.handles:
            handle.remove()


def summarize(calls):
    result = {}
    for name in sorted({key for item in calls.values() for key in item}):
        values = [value for item in calls.values() for value in item.get(name, [])]
        result[name] = {'all_ms': values, **stats(values)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    hypes = load_yaml(args.config)
    model = train_utils.create_model(hypes).cuda(args.gpu).eval()
    selected_epoch, model = train_utils.load_saved_model_diff(
        args.checkpoint_dir, model)
    model.eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False,
                        collate_fn=dataset.collate_batch_test)
    batches = []
    for batch in loader:
        if batch is not None:
            batches.append(train_utils.to_device(batch, torch.device('cuda:%d' % args.gpu)))
        if len(batches) >= args.frames:
            break

    names = ['pillar_vfe', 'scatter', 'backbone', 'shrink_conv', 'cls_head',
             'reg_head', 'dir_head', 'rain_fusion', 'fused_backbone',
             'fused_shrink_conv', 'fused_cls_head', 'fused_reg_head',
             'fused_dir_head', 'matcher']
    timer = ModuleTimer(model, names)
    box_flow_values = []
    original_generate_box_flow = model.generate_box_flow

    def timed_generate_box_flow(*call_args, **call_kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_generate_box_flow(*call_args, **call_kwargs)
        torch.cuda.synchronize()
        box_flow_values.append((time.perf_counter() - start) * 1000.0)
        return result

    model.generate_box_flow = timed_generate_box_flow
    model_ms, post_ms, wall_ms = [], [], []
    with torch.no_grad():
        for frame_index, batch in enumerate(batches):
            for _ in range(args.warmup):
                warmup_output = model(batch['ego'], dataset)
                dataset.post_process(batch, {'ego': warmup_output})
            timer.current_iter = -1
            timer.flush()
            box_flow_values.clear()
            for repeat in range(args.repeats):
                timer.current_iter = frame_index * args.repeats + repeat
                model_start = torch.cuda.Event(enable_timing=True)
                model_end = torch.cuda.Event(enable_timing=True)
                post_start = torch.cuda.Event(enable_timing=True)
                post_end = torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                model_start.record()
                output = model(batch['ego'], dataset)
                model_end.record()
                post_start.record()
                dataset.post_process(batch, {'ego': output})
                post_end.record()
                torch.cuda.synchronize(args.gpu)
                model_ms.append(float(model_start.elapsed_time(model_end)))
                post_ms.append(float(post_start.elapsed_time(post_end)))
                wall_ms.append((time.perf_counter() - start) * 1000.0)
                timer.flush()
    timer.close()

    result = {
        'config': str(Path(args.config)),
        'checkpoint_dir': str(Path(args.checkpoint_dir)),
        'gpu': args.gpu,
        'gpu_name': torch.cuda.get_device_name(args.gpu),
        'torch': torch.__version__,
        'frames': len(batches),
        'warmup_per_frame': args.warmup,
        'repeats_per_frame': args.repeats,
        'selected_epoch': selected_epoch,
        'summary_ms': {
            'end_to_end_wall': stats(wall_ms),
            'model_forward_cuda': stats(model_ms),
            'post_process_cuda': stats(post_ms),
            'generate_box_flow_wall': stats(box_flow_values),
        },
        'module_calls': summarize(timer.calls),
        'protocol': 'fixed real samples; CUDA events for model/post; explicit synchronize around generate_box_flow; no cache requested by copied config',
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
