"""Model-only latency and peak-memory profile for a registered native checkpoint.

Loads the same dataset and preprocessing as evaluation, runs one fixed batch,
and reports warmup-excluded mean/P50/P95 forward latency plus peak device
memory. Data loading is excluded; the goal is the model's own cost.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = yaml_utils.load_yaml(args.config)
    device = torch.device('cuda:0')
    model = train_utils.create_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    model.to(device).eval()
    dataset = build_dataset(cfg, visualize=False, train=False)
    timings = []
    with torch.no_grad():
        for index in range(args.warmup + args.frames):
            batch = dataset.collate_batch_test([dataset[index]])
            batch = train_utils.to_device(batch, device)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(batch['ego'])
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            if index >= args.warmup:
                timings.append(elapsed)
    timings.sort()
    result = {
        'frames': len(timings),
        'warmup': args.warmup,
        'latency_ms_mean': statistics.fmean(timings),
        'latency_ms_p50': timings[len(timings) // 2],
        'latency_ms_p95': timings[int(len(timings) * 0.95) - 1],
        'latency_ms_max': timings[-1],
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'config': args.config,
        'checkpoint': args.checkpoint,
        'scope': 'Model-only forward, batch 1, data loading and postprocess excluded',
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
