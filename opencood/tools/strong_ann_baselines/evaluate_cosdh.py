#!/usr/bin/env python
"""Official CoSDH hybrid evaluation with strict loading and auditable exports.

Run in an isolated compatible environment. No source edits and no training.
limit=0 means full split; default limit=2 is only a smoke test, never a main AP.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(str(temp), str(path))


def checkpoint_path(folder):
    best = sorted(folder.glob('net_epoch_bestval_at*.pth'))
    if best:
        if len(best) != 1:
            raise ValueError('Ambiguous best checkpoints: {}'.format(best))
        return best[0]
    numbered = []
    for path in folder.glob('net_epoch*.pth'):
        match = re.fullmatch(r'net_epoch(\d+)\.pth', path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise FileNotFoundError('No official net_epoch*.pth checkpoint in {}'.format(folder))
    return max(numbered)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', required=True, help='JSON path, absolute or relative to data-root')
    parser.add_argument('--output', required=True, help='A new output directory; existing paths rejected')
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--gpu', default='0', help='Single physical CUDA device index or UUID')
    parser.add_argument('--wire-codec',action='store_true',help='Audited demand/latent/late-box actual transport; diagnostic compute overhead included')
    parser.add_argument('--comm-range', type=float, default=None, help='Explicit participant range override in metres; omitted preserves official config')
    parser.add_argument('--compute-smoke-reference', help='Immutable full-wire reference; four-frame parity gate only')
    parser.add_argument('--compute-count',type=int,choices=(4,64),default=4)
    parser.add_argument('--compute-repeat',type=int,choices=(0,1,2),default=0)
    args = parser.parse_args()
    if args.compute_smoke_reference and (args.limit != 0 or not args.wire_codec):
        parser.error('Compute gate requires --limit 0 --wire-codec')
    compute_reference = Path(args.compute_smoke_reference).resolve() if args.compute_smoke_reference else None
    if args.limit < 0 or ',' in args.gpu:
        parser.error('limit must be >=0 and gpu must name one device')
    repo = Path(args.repo).resolve()
    model_dir = Path(args.model_dir).resolve()
    data_root = Path(args.data_root).resolve()
    split = Path(args.split)
    split = split.resolve() if split.is_absolute() else (data_root / split).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frames').mkdir()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    start_wall = time.time()
    write_json(output / 'status.json', {'stage': 'initializing', 'started_unix': start_wall, 'args': vars(args)})
    try:
        import numpy as np
        import torch
        import opencood
        from torch.utils.data import DataLoader, Subset
        from opencood.hypes_yaml import yaml_utils
        from opencood.tools import train_utils, inference_utils
        from opencood.data_utils.datasets import build_dataset
        from opencood.utils import eval_utils

        package_file = Path(opencood.__file__).resolve()
        if repo not in package_file.parents:
            raise RuntimeError('Wrong opencood package: {}'.format(package_file))
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for this benchmark; CPU fallback is forbidden')
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.25, device=0)
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(303)
        np.random.seed(303)
        torch.manual_seed(303)
        torch.cuda.manual_seed_all(303)
        device = torch.device('cuda:0')

        config_path = model_dir / 'config.yaml'
        cfg = yaml_utils.load_yaml(str(config_path))
        # Only dataset locations and the explicitly requested official inference mode.
        overrides = {'data_dir': str(data_root), 'root_dir': str(data_root / 'train.json'),
                     'validate_dir': str(split), 'test_dir': str(split)}
        if args.comm_range is not None:
            if not np.isfinite(args.comm_range) or args.comm_range <= 0:
                raise ValueError('comm-range must be finite and positive')
            overrides['comm_range'] = args.comm_range
        cfg.update(overrides)
        original_fusion = cfg['fusion']['core_method']
        cfg['fusion']['core_method'] = 'intermediatelate'
        if 'box_align' in cfg:
            raise RuntimeError('Unexpected box_align external files: resolve explicitly before reproduction')
        checkpoint = checkpoint_path(model_dir)
        model = train_utils.create_model(cfg)
        saved = torch.load(str(checkpoint), map_location='cpu')
        expected = model.state_dict()
        if not isinstance(saved, dict):
            raise TypeError('Expected official raw state_dict checkpoint')
        missing = sorted(set(expected) - set(saved))
        unexpected = sorted(set(saved) - set(expected))
        shape_mismatches = {k: {'expected': list(expected[k].shape),
                                'actual': list(saved[k].shape) if hasattr(saved[k], 'shape') else 'not tensor'}
                            for k in set(expected) & set(saved)
                            if not isinstance(saved[k], torch.Tensor) or expected[k].shape != saved[k].shape}
        load_audit = {'missing': missing, 'unexpected': unexpected, 'shape_mismatches': shape_mismatches}
        write_json(output / 'load_audit.json', load_audit)
        if missing or unexpected or shape_mismatches:
            raise RuntimeError('Strict checkpoint audit failed; see load_audit.json')
        if any(not torch.isfinite(v).all().item() for v in saved.values() if torch.is_floating_point(v)):
            raise ValueError('Nonfinite checkpoint tensor')
        model.load_state_dict(saved, strict=True)
        del saved
        model.to(device).eval()
        parameters = sum(p.numel() for p in model.parameters())

        dataset = build_dataset(cfg, visualize=True, train=False)
        wire=None
        if args.wire_codec:
            from cosdh_wire import CoSDHWire
            wire=CoSDHWire(model,dataset,output)
        ids = list(dataset.split_info)
        source_ids = json.loads(split.read_text(encoding='utf-8'))
        if ids != source_ids or len(ids) != len(dataset):
            raise RuntimeError('Dataset did not preserve requested ordered split')
        count = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
        if count == 0:
            raise RuntimeError('Empty evaluation split')
        loader = DataLoader(Subset(dataset, list(range(count))), batch_size=1, num_workers=0,
                            collate_fn=dataset.collate_batch_test, shuffle=False,
                            pin_memory=False, drop_last=False)
        try:
            commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
        except Exception:
            commit = None
        metadata = {'repo': str(repo), 'repo_commit': commit, 'opencood_file': str(package_file),
                    'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                    'source_config': str(config_path), 'source_config_sha256': digest(config_path),
                    'split': str(split), 'split_sha256': digest(split), 'ordered_split_ids': ids,
                    'data_root': str(data_root), 'overrides': overrides,
                    'original_fusion': original_fusion, 'fusion_method': 'intermediatelate',
                    'limit': args.limit, 'evaluated_samples_planned': count, 'dataset_samples': len(dataset),
                    'metric': 'official BEV polygon IoU AP with global prediction score sort',
                    'gt_parity_with_our_snn': 'NOT_ESTABLISHED', 'actual_communication_bytes': None,
                    'seed': 303, 'parameters': parameters, 'torch_version': torch.__version__,
                    'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                    'physical_gpu_selection': args.gpu, 'allocator_fraction': .25,
                    'warmup_forwards_first_batch': 3,
                    'timing_scope': 'CUDA-synchronized official hybrid inference incl both model forwards and postprocess/NMS, excludes loading/H2D/export/AP; batch1',
                    'source_script_sha256': digest(__file__), 'started_unix': start_wall}
        write_json(output / 'provenance.json', metadata)
        metadata['wire_codec']='demand_sparse_latent_dependency_closure_late_boxes_v1' if args.wire_codec else None
        metadata['wire_compute_scope']='includes redundant diagnostic decode/warp, not optimized algorithmic cost' if args.wire_codec else None
        write_json(output / 'provenance.json', metadata)
        if compute_reference:
            from compute_cosdh_adapter import smoke
            try:smoke(model,dataset,wire,ids,compute_reference,output,metadata,args.gpu,loader,args.compute_count,args.compute_repeat)
            finally:wire.close()
            return
        stat = {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in (.3, .5, .7)}
        rows = []
        # Store output references, and inspect only AFTER timing to avoid finite-check synchronization overhead.
        raw_outputs = []
        hook = model.register_forward_hook(lambda module, inputs, result: raw_outputs.append(result))

        def finite_tree(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all().item())
            if isinstance(value, dict):
                return all(finite_tree(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite_tree(v) for v in value)
            return True

        def array(tensor, empty_shape):
            return np.empty(empty_shape, dtype=np.float32) if tensor is None else tensor.detach().cpu().numpy()

        iterator = iter(loader)
        with torch.no_grad():
            for index in range(count):
                t0 = time.perf_counter()
                batch = next(iterator)
                load_seconds = time.perf_counter() - t0
                if batch is None:
                    raise RuntimeError('Dataset returned None at {}'.format(index))
                if int(batch['ego']['sample_idx']) != index:
                    raise RuntimeError('Frame index mismatch')
                t0 = time.perf_counter()
                batch = train_utils.to_device(batch, device)
                if wire:wire.begin(batch,str(ids[index]))
                torch.cuda.synchronize()
                h2d_seconds = time.perf_counter() - t0
                if index == 0:
                    for _ in range(3):
                        raw_outputs.clear()
                        if wire:wire.begin(batch,str(ids[index]))
                        warmup = inference_utils.inference_late_fusion(batch, model, dataset)
                        if not finite_tree(warmup) or not finite_tree(raw_outputs):
                            raise ValueError('Nonfinite warmup output')
                    del warmup
                    raw_outputs.clear()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                raw_outputs.clear()
                if wire:wire.begin(batch,str(ids[index]))
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = inference_utils.inference_late_fusion(batch, model, dataset)
                torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - t0
                if not finite_tree(result) or not finite_tree(raw_outputs):
                    raise ValueError('Nonfinite model/output at {}'.format(ids[index]))
                pred = array(result['pred_box_tensor'], (0, 8, 3))
                score = array(result['pred_score'], (0,))
                gt = array(result['gt_box_tensor'], (0, 8, 3))
                if pred.shape != (len(score), 8, 3) or gt.shape[1:] != (8, 3):
                    raise ValueError('Unexpected box/score shape')
                for threshold in stat:
                    eval_utils.caluclate_tp_fp(result['pred_box_tensor'], result['pred_score'],
                                              result['gt_box_tensor'], stat, threshold)
                frame_path = output / 'frames' / ('{:06d}.npz'.format(index))
                np.savez_compressed(str(frame_path), frame_id=np.asarray(str(ids[index])),
                                    pred=pred, score=score, gt=gt)
                row = {'index': index, 'frame_id': str(ids[index]), 'prediction_count': len(score),
                       'gt_count': len(gt), 'file': str(frame_path.relative_to(output)),
                       'sha256': digest(frame_path), 'inference_seconds': inference_seconds,
                       'data_loading_seconds': load_seconds, 'h2d_seconds': h2d_seconds,
                       'model_forwards': len(raw_outputs),
                       'record_len': batch['ego']['record_len'].detach().cpu().tolist(),
                       'gt_array_sha256': hashlib.sha256(gt.tobytes(order='C')).hexdigest()}
                rows.append(row)
                if wire:row['wire']=wire.finish(index)
                with (output / 'frames.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, allow_nan=False) + '\n')
                if (index + 1) % 25 == 0 or index == count - 1:
                    write_json(output / 'status.json', {'stage': 'evaluating', 'completed': index + 1,
                                                       'planned': count, 'last_frame_id': str(ids[index]),
                                                       'updated_unix': time.time()})
                    print('completed {}/{}'.format(index + 1, count), flush=True)
        hook.remove()
        if wire:wire.close()
        ap = {str(t): float(eval_utils.calculate_ap(stat, t)[0]) for t in stat}
        timings = np.asarray([r['inference_seconds'] for r in rows])
        summary = {'status': 'completed', 'full_split': count == len(dataset),
                   'smoke_only': count < len(dataset), 'samples': count,
                   'gt_boxes': stat[.7]['gt'], 'pred_boxes': sum(r['prediction_count'] for r in rows),
                   'bev_ap': ap, 'parameters': parameters,
                   'inference_ms_mean': float(timings.mean() * 1000),
                   'inference_ms_median': float(np.median(timings) * 1000),
                   'inference_ms_p95': float(np.percentile(timings, 95) * 1000),
                   'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                   'peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                   'all_outputs_finite': True, 'actual_communication_bytes': None,
                   'gt_parity_with_our_snn': 'NOT_ESTABLISHED',
                   'frames_manifest_sha256': digest(output / 'frames.jsonl'),
                   'completed_unix': time.time(), 'wall_seconds': time.time() - start_wall}
        write_json(output / 'result_stat.json', stat)
        if wire:
            def stats(values):
                return {'mean':float(np.mean(values)),'p95':float(np.percentile(values,95)),'max':float(np.max(values)),'count':len(values)}
            summary['wire_statistics']={k:stats([r['wire'][k] for r in rows]) for k in ('demand_bytes','response_bytes','total_bytes','envelope_bytes','late_boxes')}
            summary['actual_communication_bytes']=summary['wire_statistics']['total_bytes']['mean']
            summary['wire_note']='Actual binary demand, three dependency-closed FP16 latent/mask sections and pre-NMS late boxes; 160-byte two-way metadata included; link/IP overhead excluded; diagnostic redundant compute not suitable for efficiency comparison'
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', summary)
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    except Exception:
        write_json(output / 'failure.json', {'failed_unix': time.time(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
