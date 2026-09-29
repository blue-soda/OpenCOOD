"""Bounded trained-feature audit: sparsity, actual packet bytes, exact replay.

This is a single-agent lossless codec test, not a cooperative AP experiment.
No wireless transport, geometry alignment, latency or energy claim is made.
"""
import argparse
import hashlib
import json
import random
import struct
import subprocess
import time
from pathlib import Path
import zlib
import numpy as np
import torch
import spconv.pytorch as spconv
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import write_json
from opencood.utils.spike_packet import HEADER, encode, decode


def reference_packet(coords, values, metadata, dtype):
    meta = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    payload = coords.astype('<u2').tobytes() + values.astype(dtype).tobytes()
    body = HEADER.pack(b'REF0', 1, 2, values.shape[1], len(values), len(meta), len(payload)) + meta + payload
    return body + struct.pack('<I', zlib.crc32(body) & 0xffffffff)


def run(args, folder):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = load_yaml(args.config)
    dataset = build_dataset(cfg, train=False)
    if cfg['fusion']['core_method'] != 'SingleDAIRVehicle':
        raise ValueError('Audit expects the fixed single-agent protocol')
    if not 0 < args.samples <= len(dataset):
        raise ValueError('Invalid sample count')
    indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=int).tolist()
    checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    model = train_utils.create_model(cfg).cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    manifest = {'arguments': vars(args), 'checkpoint_sha256': checkpoint_hash,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        'data_manifest_sha256': hashlib.sha256(Path(cfg['data_manifest']).read_bytes()).hexdigest(),
        'indices': indices, 'frame_ids': [dataset.data[i] for i in indices],
        'protocol': '64 evenly spaced validation frames by default; no selection by AP or sparsity',
        'metadata_limit': 'sender-local index contract only; physical feature centers, pose/timestamp and network framing not implemented'}
    write_json(folder / 'manifest.json', manifest)
    rows = []
    started = time.monotonic()
    for index in indices:
        if time.monotonic() - started > args.timeout_seconds:
            raise TimeoutError('Bounded packet audit time limit')
        batch = train_utils.to_device(dataset.collate_batch_test([dataset[index]]), torch.device('cuda'))
        capture = {}
        def capture_encoded(module, inputs, output):
            capture['tensor'] = output['encoded_spconv_tensor']
        hook = model.backbone_3d.register_forward_hook(capture_encoded)
        try:
            with torch.no_grad():
                original_output = model(batch['ego'])
        finally:
            hook.remove()
        original = capture['tensor']
        if original.batch_size != 1 or not torch.all(original.indices[:, 0] == 0):
            raise ValueError('A packet represents one sender, not a batch')
        coords = original.indices[:, 1:].detach().cpu().numpy()
        values = original.features.detach().cpu().numpy()
        metadata = {'layer': 'backbone_3d.conv_out', 'frame_id': dataset.data[index],
            'model_sha256': checkpoint_hash, 'spatial_shape_zyx': list(map(int, original.spatial_shape)),
            'coordinate_contract': 'sender-local sparse tensor indices; physical alignment pending'}
        lengths = {}
        for mode in ('dense3', 'mask3'):
            lengths[mode] = len(encode(coords, values, metadata, mode))
        begin = time.perf_counter()
        packet = encode(coords, values, metadata, 'auto')
        encode_ms = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        decoded_coords, decoded_values, decoded_metadata = decode(packet)
        decode_ms = (time.perf_counter() - begin) * 1000
        keep = np.any(values != 0, axis=1)
        np.testing.assert_array_equal(decoded_coords, coords[keep])
        np.testing.assert_array_equal(decoded_values, values[keep])
        assert metadata == decoded_metadata
        restored_coords = np.column_stack([np.zeros(len(decoded_coords), dtype=np.int32), decoded_coords]).astype(np.int32)
        restored = spconv.SparseConvTensor(torch.from_numpy(decoded_values.astype(np.float32)).cuda(),
            torch.from_numpy(restored_coords).cuda(), original.spatial_shape, 1)
        assert torch.equal(original.dense(), restored.dense()), 'Dense features changed after codec'
        def replay_encoded(module, inputs, output):
            return dict(output, encoded_spconv_tensor=restored)
        hook = model.backbone_3d.register_forward_hook(replay_encoded)
        try:
            with torch.no_grad():
                replay_output = model(batch['ego'])
        finally:
            hook.remove()
        for key in ('psm', 'rm', 'dm'):
            assert torch.equal(original_output[key], replay_output[key]), 'Detector output mismatch: ' + key
        with torch.no_grad():
            original_boxes = dataset.post_process_no_fusion(batch, {'ego': original_output})[:2]
            replay_boxes = dataset.post_process_no_fusion(batch, {'ego': replay_output})[:2]
        for a, b in zip(original_boxes, replay_boxes):
            assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b)), 'Decoded detections changed'
        row = {'frame_id': dataset.data[index], 'active_rows': len(values), 'nonzero_rows': int(keep.sum()),
            'spatial_cells': int(np.prod(original.spatial_shape)), 'channels': values.shape[1],
            'count_histogram': np.bincount(values.astype(np.int64).ravel(), minlength=5).tolist(),
            'packet_bytes': len(packet), 'dense3_bytes': lengths['dense3'], 'mask3_bytes': lengths['mask3'],
            'codec_mode': 'dense3' if packet[5] == 0 else 'mask3',
            'raw_fp32_packet_bytes': len(reference_packet(coords, values, metadata, '<f4')),
            'pruned_fp32_packet_bytes': len(reference_packet(coords[keep], values[keep], metadata, '<f4')),
            'pruned_uint8_packet_bytes': len(reference_packet(coords[keep], values[keep], metadata, 'u1')),
            'encode_cpu_ms': encode_ms, 'decode_cpu_ms': decode_ms, 'exact_features_and_predictions': True}
        if not rows:
            (folder / 'first_packet.bin').write_bytes(packet)
            np.savez_compressed(str(folder / 'first_features.npz'), coords=coords, features=values)
        rows.append(row)
        with (folder / 'frames.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if len(rows) % 8 == 0:
            print(json.dumps({'completed': len(rows), 'total': len(indices)}), flush=True)
    hist = np.sum([r['count_histogram'] for r in rows], axis=0)
    totals = {key: sum(r[key] for r in rows) for key in ('active_rows', 'nonzero_rows', 'spatial_cells',
        'packet_bytes', 'raw_fp32_packet_bytes', 'pruned_fp32_packet_bytes', 'pruned_uint8_packet_bytes')}
    summary = {'passed': True, 'samples': len(rows), 'checkpoint_sha256': checkpoint_hash,
        'all_features_heads_boxes_scores_exact': True, 'count_histogram': hist.tolist(), 'totals': totals,
        'value_zero_fraction': float(hist[0] / hist.sum()),
        'all_zero_row_fraction': 1. - totals['nonzero_rows'] / totals['active_rows'],
        'active_spatial_fraction': totals['active_rows'] / totals['spatial_cells'],
        'nonzero_spatial_fraction': totals['nonzero_rows'] / totals['spatial_cells'],
        'compression_vs_raw_fp32': totals['raw_fp32_packet_bytes'] / totals['packet_bytes'],
        'compression_vs_pruned_fp32': totals['pruned_fp32_packet_bytes'] / totals['packet_bytes'],
        'compression_vs_pruned_uint8': totals['pruned_uint8_packet_bytes'] / totals['packet_bytes'],
        'packet_bytes_mean': totals['packet_bytes'] / len(rows),
        'packet_bytes_p95': float(np.percentile([r['packet_bytes'] for r in rows], 95)),
        'codec_mode_counts': {mode: sum(r['codec_mode'] == mode for r in rows) for mode in ('dense3', 'mask3')},
        'cpu_timing_ms': {key: {'median': float(np.median([r[key] for r in rows])),
                               'p95': float(np.percentile([r[key] for r in rows], 95))}
                          for key in ('encode_cpu_ms', 'decode_cpu_ms')},
        'timing_limit': 'CPU prototype only, shared server load, excludes GPU transfers, fragmentation and transport',
        'scope': 'lossless single-agent Count4 codec; no cooperative AP, ANN superiority, hardware energy or bandwidth claim',
        'seconds': time.monotonic() - started}
    write_json(folder / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='opencood/hypes_yaml/dair-v2x/snn/e3dsnn_vehicle_single.yaml')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--timeout-seconds', type=int, default=1200)
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        run(args, folder)
        write_json(folder / 'status.json', {'state': 'complete'})
    except Exception:
        import traceback
        write_json(folder / 'status.json', {'state': 'failed', 'error': traceback.format_exc()})
        raise
