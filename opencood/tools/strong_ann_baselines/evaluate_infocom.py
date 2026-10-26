"""Strict, standalone official InfoCom evaluation; does not modify its checkout.

Run from any directory with an isolated environment and --repo pointing to the
official checkout. --limit 0 evaluates every sample, default 2 is only a smoke.
Communication bytes are deliberately NOT estimated as a real packet metric.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False) + '\n', encoding='utf-8')
    os.replace(str(tmp), str(path))


def select_checkpoint(model_dir):
    best = sorted(model_dir.glob('net_epoch_bestval_at*.pth'))
    if best:
        if len(best) != 1:
            raise ValueError('Ambiguous best checkpoint: ' + str(best))
        return best[0]
    numbered = []
    for path in model_dir.glob('net_epoch*.pth'):
        match = re.fullmatch(r'net_epoch(\d+)\.pth', path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise ValueError('No official net_epoch*.pth checkpoint in model-dir')
    return max(numbered, key=lambda pair: pair[0])[1]


def run(args, output):
    # Set visibility before importing Torch or any official module.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    import opencood
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils, inference_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils import eval_utils
    from torch.utils.data import DataLoader, Subset

    if repo not in Path(opencood.__file__).resolve().parents:
        raise RuntimeError('Wrong opencood imported: ' + str(opencood.__file__))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this evaluation protocol')
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    random.seed(303)
    np.random.seed(303)
    torch.manual_seed(303)
    torch.cuda.manual_seed_all(303)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_dir = Path(args.model_dir).resolve()
    config_path = model_dir / 'config.yaml'
    split_path = Path(args.split).resolve()
    data_root = Path(args.data_root).resolve()
    ids = json.loads(split_path.read_text(encoding='utf-8'))
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError('Split must be a nonempty unique list of official IDs')
    hypes = yaml_utils.load_yaml(str(config_path))
    hypes.update(data_dir=str(data_root), root_dir=str(split_path),
                 validate_dir=str(split_path), test_dir=str(split_path))
    if args.comm_range is not None:
        if not np.isfinite(args.comm_range) or args.comm_range <= 0:
            raise ValueError('comm-range must be finite and positive')
        hypes['comm_range'] = args.comm_range
    if hypes['fusion']['core_method'] != 'intermediate':
        raise ValueError('Expected official intermediate fusion config')
    if hypes['fusion']['dataset'].lower() != 'dairv2x':
        raise ValueError('Expected DAIR-V2X official config')
    checkpoint = select_checkpoint(model_dir)
    model = train_utils.create_model(hypes)
    # Only load explicitly trusted official/local checkpoints, not arbitrary files.
    loaded = torch.load(str(checkpoint), map_location='cpu')
    wrapper = None
    if isinstance(loaded, dict) and 'state_dict' in loaded:
        wrapper = 'state_dict'
        loaded = loaded['state_dict']
    if not isinstance(loaded, dict):
        raise ValueError('Checkpoint is not a state dictionary')
    expected = model.state_dict()
    missing = sorted(set(expected) - set(loaded))
    unexpected = sorted(set(loaded) - set(expected))
    raw_mismatch = {'missing_keys': missing, 'unexpected_keys': unexpected}
    compatibility = None
    if args.checkpoint_compat:
        required_missing = ['ib_communication.mu_net.1.scale', 'ib_communication.mu_net.1.shift',
                            'ib_communication.std_net.0.bias', 'ib_communication.std_net.0.weight',
                            'ib_communication.std_net.1.scale', 'ib_communication.std_net.1.shift']
        required_extra = ['ib_communication.sigma_net.0.bias', 'ib_communication.sigma_net.0.weight']
        if missing != required_missing or unexpected != required_extra:
            raise ValueError('Legacy compatibility is limited to the exact published mismatch')
        for suffix in ('weight', 'bias'):
            loaded['ib_communication.std_net.0.' + suffix] = loaded.pop('ib_communication.sigma_net.0.' + suffix)
        for layer in ('mu_net', 'std_net'):
            for suffix, value in (('scale', 1.), ('shift', 0.)):
                key = 'ib_communication.' + layer + '.1.' + suffix
                assert torch.all(expected[key] == value)
                loaded[key] = torch.full_like(expected[key], value)
        compatibility = 'legacy sigma->std; newly added AdaptiveScale layers set to identity; inference only, not proof of original training architecture'
        missing = sorted(set(expected) - set(loaded))
        unexpected = sorted(set(loaded) - set(expected))
    shape_errors = {key: {'expected': list(expected[key].shape),
                         'found': list(loaded[key].shape) if hasattr(loaded[key], 'shape') else None}
                    for key in set(expected).intersection(loaded)
                    if not hasattr(loaded[key], 'shape') or loaded[key].shape != expected[key].shape}
    load_audit = {'checkpoint': str(checkpoint), 'sha256': sha256(checkpoint),
                  'wrapper': wrapper, 'expected_keys': len(expected),
                  'loaded_keys': len(loaded), 'missing_keys': missing,
                  'unexpected_keys': unexpected, 'shape_errors': shape_errors,
                  'strict_passed': not (missing or unexpected or shape_errors)}
    load_audit.update(raw_checkpoint_mismatch=raw_mismatch, explicit_compatibility=compatibility)
    write_json(output / 'load_audit.json', load_audit)
    if not load_audit['strict_passed']:
        raise ValueError('Checkpoint mismatch; see load_audit.json; no partial load allowed')
    model.load_state_dict(loaded, strict=True)
    nonfinite = [key for key, tensor in model.state_dict().items()
                 if tensor.is_floating_point() and not torch.isfinite(tensor).all().item()]
    if nonfinite:
        raise ValueError('Nonfinite checkpoint tensors: ' + str(nonfinite))
    del loaded, expected
    parameters = {name: sum(p.numel() for p in module.parameters())
                  for name, module in model.named_children()}
    parameter_count = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model = model.cuda().eval()
    mask_ratio = getattr(args, 'mask_ratio', None)
    if mask_ratio is not None:
        # Frozen-checkpoint operating curve: retain the official learned raw
        # mask ranking, tie handling and quantization/STE expression exactly.
        # This changes only the requested top-k spatial fraction, not weights.
        namespace = model.ib_communication.get_mask.__func__.__globals__
        original_topk = namespace['topk_mask']
        def budget_topk(mask, ratio=.1):
            return original_topk(mask, ratio=mask_ratio)
        namespace['topk_mask'] = budget_topk
    dataset = build_dataset(hypes, visualize=True, train=False)
    if list(dataset.split_info) != ids or len(dataset) != len(ids):
        raise ValueError('Actual dataset does not match requested ordered IDs')
    count = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    selected = Subset(dataset, list(range(count)))
    loader = DataLoader(selected, batch_size=1, num_workers=0,
                        collate_fn=dataset.collate_batch_test, shuffle=False,
                        pin_memory=False, drop_last=False)
    try:
        git_commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        git_commit = None
    meta = {'repo': str(repo), 'repo_commit': git_commit,
            'script_sha256': sha256(Path(__file__)), 'model_dir': str(model_dir),
            'config_sha256': sha256(config_path), 'checkpoint_sha256': load_audit['sha256'],
            'data_root': str(data_root), 'split': str(split_path),
            'split_sha256': sha256(split_path), 'total_split_count': len(ids),
            'requested_limit': args.limit, 'evaluated_ids': ids[:count],
            'seed': 303, 'batch_size': 1, 'num_workers': 0,
            'gpu_selector': args.gpu, 'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_fraction': .25, 'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda, 'numpy_version': np.__version__,
            'parameter_count': parameter_count, 'parameter_bytes': parameter_bytes,
            'parameters_by_top_module': parameters,
            'metric': 'official BEV polygon IoU AP, global confidence sorting',
            'wire_bytes': None, 'wire_note': 'Official tensor path has no real serialized codec',
            'warmup_passes': 3,
            'timing_scope': 'GPU-synchronized model forward and official helper incl CPU postprocess; excludes data loading, H2D, metric calculation, file IO',
            'overrides': {'data_dir': str(data_root), 'root_dir': str(split_path),
                          'validate_dir': str(split_path), 'test_dir': str(split_path)}}
    write_json(output / 'metadata.json', meta)
    meta['participant_range_metres'] = hypes['comm_range']
    meta['comm_range_override'] = args.comm_range
    meta['mask_ratio_override'] = mask_ratio
    meta['mask_ratio_note'] = ('Frozen official checkpoint operating point; not budget-adapted training; '
        'uses official raw learned ranking, ties, quantization and decoder; actual bytes measured, no hard byte cap') if mask_ratio is not None else None
    adaptation_protocol=getattr(args,'adaptation_protocol',None)
    if adaptation_protocol is not None:
        protocol_path=Path(adaptation_protocol).resolve()
        protocol=json.loads(protocol_path.read_text())
        if args.checkpoint_compat:
            raise ValueError('Adapted checkpoints must already contain canonical strict keys')
        meta['adaptation_protocol']={'file':str(protocol_path),'sha256':sha256(protocol_path),'contract':protocol}
        meta['mask_ratio_note']='Adapted checkpoint operating point; actual bytes measured, no hard byte cap'
    meta['checkpoint_compatibility'] = compatibility
    meta['wire_codec']='exact_fp32_bitmap_dictionary_or_dense_zlib_v1' if args.wire_codec else None
    write_json(output / 'metadata.json', meta)
    wire=None
    if args.wire_codec:
        from infocom_wire import WireAudit
        wire=WireAudit(model,output)
    compute_reference = getattr(args, 'compute_smoke_reference', None)
    if compute_reference is not None:
        from compute_infocom_adapter import smoke
        try:
            smoke(model, dataset, wire, ids, Path(compute_reference), output, meta, args.gpu, loader)
        finally:
            if wire:wire.handle.remove()
        return
    # Hooks time only the unchanged official model forward. The public official
    # helper performs this forward then unchanged dataset.post_process.
    timing = {}
    def before(module, inputs):
        torch.cuda.synchronize()
        timing['start'] = time.perf_counter()
    def after(module, inputs, result):
        torch.cuda.synchronize()
        timing.setdefault('durations', []).append(time.perf_counter() - timing['start'])
    h1 = model.register_forward_pre_hook(before)
    h2 = model.register_forward_hook(after)
    stat = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for iou in (.3, .5, .7)}
    index = []
    prediction_dir = output / 'predictions'
    prediction_dir.mkdir()
    def array(tensor, empty_shape):
        if tensor is None:
            return np.empty(empty_shape, dtype=np.float32)
        result = tensor.detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite predictions, scores or GT')
        return result
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                raise ValueError('Dataset returned None for ' + str(ids[i]))
            if int(batch['ego']['sample_idx']) != i:
                raise ValueError('Sample index mismatch')
            batch = train_utils.to_device(batch, torch.device('cuda:0'))
            if wire:wire.begin(batch,str(ids[i]))
            if i == 0:
                # Warmup must not consume randomness of the scored pass.
                states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                          torch.cuda.get_rng_state_all())
                for unused in range(3):
                    inference_utils.inference_intermediate_fusion(batch, model, dataset)
                random.setstate(states[0]); np.random.set_state(states[1])
                torch.set_rng_state(states[2]); torch.cuda.set_rng_state_all(states[3])
                timing['durations'] = []
                if wire:wire.begin(batch,str(ids[i]))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = inference_utils.inference_intermediate_fusion(batch, model, dataset)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if len(timing['durations']) != i + 1:
                raise ValueError('Unexpected number of model forward calls')
            pred, score, gt = (result['pred_box_tensor'], result['pred_score'], result['gt_box_tensor'])
            pred_np, score_np, gt_np = array(pred, (0, 8, 3)), array(score, (0,)), array(gt, (0, 8, 3))
            if len(pred_np) != len(score_np):
                raise ValueError('Prediction/score length mismatch')
            for iou in stat:
                eval_utils.caluclate_tp_fp(pred, score, gt, stat, iou)
            target = prediction_dir / ('%06d.npz' % i)
            np.savez_compressed(str(target), frame_id=np.asarray(str(ids[i])),
                                pred=pred_np, score=score_np, gt=gt_np)
            index.append({'index': i, 'frame_id': str(ids[i]), 'file': 'predictions/' + target.name,
                          'sha256': sha256(target), 'pred_count': len(pred_np), 'gt_count': len(gt_np),
                          'record_len': batch['ego']['record_len'].detach().cpu().tolist(),
                          'forward_seconds': timing['durations'][-1],
                          'forward_postprocess_seconds': elapsed})
            if wire:index[-1]['wire']=wire.finish(i)
            write_json(output / 'status.json', {'processed': i + 1, 'planned': count,
                                               'frame_id': str(ids[i]), 'finished': False})
            print(json.dumps(index[-1]), flush=True)
            del batch, result, pred, score, gt
    h1.remove(); h2.remove()
    if wire:wire.handle.remove()
    ap = eval_utils.eval_final_results(stat, str(output), 'official')
    def statistics(values):
        return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
                'p95': float(np.percentile(values, 95)), 'min': float(np.min(values)),
                'max': float(np.max(values)), 'count': len(values)}
    summary = dict(meta)
    summary.update({'completed': True, 'full_split': count == len(ids), 'frames': count,
                    'gt_count': sum(x['gt_count'] for x in index),
                    'pred_count': sum(x['pred_count'] for x in index),
                    'ap30': float(ap[0]), 'ap50': float(ap[1]), 'ap70': float(ap[2]),
                    'forward_seconds': statistics([x['forward_seconds'] for x in index]),
                    'forward_postprocess_seconds': statistics([x['forward_postprocess_seconds'] for x in index]),
                    'cuda_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                    'cuda_peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                    'timing_is_smoke_only': count < 30})
    if wire:
        summary['wire_statistics']={k:statistics([x['wire'][k] for x in index]) for k in
                                    ('packet_bytes','envelope_bytes','total_bytes','encode_seconds','decode_seconds')}
        summary['wire_exact_fp32']=True
        summary['wire_note']='Measured exact codec; separately counted 80-byte frame/sender/sequence/pairwise-transform envelope; excludes link/IP overhead. CPU codec included in forward timing.'
    summary['participant_counts']={str(n):sum(x['record_len']==[n] for x in index) for n in (1,2)}
    write_json(output / 'predictions_index.json', index)
    write_json(output / 'summary.json', summary)
    write_json(output / 'status.json', {'processed': count, 'planned': count, 'finished': True})
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'model-dir', 'data-root', 'split', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--comm-range', type=float, default=None)
    parser.add_argument('--mask-ratio',type=float,default=None,
        help='Frozen-model top-k mask fraction in (0,0.1]; explicit operating-curve override, not a byte cap')
    parser.add_argument('--checkpoint-compat', action='store_true', help='Explicit legacy sigma/identity-scale inference mapping; exact mismatch only')
    parser.add_argument('--wire-codec',action='store_true',help='Serialize, decode and use every non-ego latent/mask; exact FP32')
    parser.add_argument('--adaptation-protocol',help='Training protocol for an adapted canonical checkpoint; records provenance')
    parser.add_argument('--compute-smoke-reference', help='Four fixed-frame real-wire timing adapter parity gate; not an AP/performance run')
    args = parser.parse_args()
    if args.mask_ratio is not None and (not math.isfinite(args.mask_ratio) or not 0 < args.mask_ratio <= .1):
        parser.error('mask ratio must be finite and in (0,0.1]')
    if args.limit < 0 or ',' in args.gpu:
        parser.error('limit must be nonnegative; select exactly one GPU')
    if args.compute_smoke_reference and not args.wire_codec:
        parser.error('Compute smoke requires actual wire codec')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(args, output)
    except Exception:
        write_json(output / 'failure.json', {'completed': False, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
