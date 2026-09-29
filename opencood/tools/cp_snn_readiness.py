"""Bounded real-data readiness audit, NOT an E-3DSNN detector or AP evaluation.

The independent count operator follows floor(clamp(x, 0, 4) + .5), with
the (0, 4) straight-through gradient used by E-3DSNN's Multispike:
https://github.com/bollossom/E-3DSNN/blob/dbe5d1731d3204850bd591a18f7255ca2d1b6712/det/pcdet/models/backbones_3d/spconv_backbone_spike.py
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import traceback

import numpy as np
import torch
from torch import nn


class Count4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        ctx.save_for_backward(values)
        return torch.floor(values.clamp(0, 4) + 0.5)

    @staticmethod
    def backward(ctx, grad):
        values, = ctx.saved_tensors
        return grad * ((values > 0) & (values < 4)).to(grad.dtype)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sensor_index(root, side):
    with open(str(root / side / 'data_info.json')) as stream:
        return {Path(row['pointcloud_path']).stem: row for row in json.load(stream)
                if row.get('pointcloud_path')}


def sparse_probe(points, lidar_range, voxel_size):
    import spconv.pytorch as spconv
    from opencood.data_utils.pre_processor.sp_voxel_preprocessor import Spconv2VoxelGenerator

    generator = Spconv2VoxelGenerator(voxel_size, lidar_range, 32, 70000)
    coords, features, agent_stats = [], [], []
    for agent, cloud in enumerate(points):
        voxels, indices, numbers = generator.generate(cloud[:, :4])
        if not len(indices):
            raise ValueError('Empty 3D voxel cloud for agent {}'.format(agent))
        coords.append(np.column_stack((np.full(len(indices), agent), indices)))
        features.append(voxels.sum(axis=1) / numbers[:, None])
        agent_stats.append({'raw_points': len(cloud), 'active_voxels': len(indices),
                            'voxel_cap_reached': len(indices) == 70000})
    indices = torch.as_tensor(np.concatenate(coords), dtype=torch.int32, device='cuda')
    values = torch.as_tensor(np.concatenate(features), dtype=torch.float32, device='cuda')
    values.requires_grad_(True)
    grid = np.rint((np.array(lidar_range[3:]) - lidar_range[:3]) / voxel_size).astype(int)
    tensor = spconv.SparseConvTensor(values, indices, grid[::-1].tolist(), len(points))
    # Random, training-mode diagnostic stem; this is not the paper backbone.
    stem = spconv.SubMConv3d(4, 16, 3, padding=1, bias=False).cuda()
    norm = nn.BatchNorm1d(16, eps=1e-3, momentum=0.01).cuda().train()
    encoded = stem(tensor)
    counts = Count4.apply(norm(encoded.features))
    assert torch.equal(encoded.indices, indices)
    assert torch.isfinite(counts).all() and torch.equal(counts, counts.round())
    assert counts.min() >= 0 and counts.max() <= 4
    counts.square().mean().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0
    assert torch.isfinite(stem.weight.grad).all() and stem.weight.grad.abs().sum() > 0
    torch.cuda.synchronize()
    histogram = torch.bincount(counts.detach().long().flatten(), minlength=5).cpu().tolist()
    return {'agents': agent_stats, 'grid_xyz': grid.tolist(), 'feature_shape': list(counts.shape),
            'count_histogram_0_to_4': histogram, 'nonzero_value_fraction': float((counts != 0).float().mean()),
            'input_grad_abs_sum': float(values.grad.abs().sum()),
            'weight_grad_abs_sum': float(stem.weight.grad.abs().sum()),
            'count_tensor_bytes_float32': counts.numel() * counts.element_size(),
            'coordinate_tensor_bytes_int32': indices.numel() * indices.element_size(),
            'packed_payload_bytes': None,
            'note': 'No codec, fusion, detector head, trained weights, or energy measurement.'}


def run(args, report):
    import spconv
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils.transformation_utils import x1_to_x2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the sparse forward/backward audit')
    report['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'spconv': spconv.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
        'python_executable': os.sys.executable}
    boundary = torch.tensor([-1., 0., .49, .5, 1.5, 3.5, 4., 5.], device='cuda', requires_grad=True)
    counted = Count4.apply(boundary)
    assert counted.tolist() == [0., 0., 0., 1., 2., 4., 4., 4.]
    counted.sum().backward()
    assert boundary.grad.tolist() == [0., 0., 1., 1., 1., 1., 0., 0.]
    report['count_operator_boundary_check'] = 'passed'
    cfg = load_yaml(args.config)
    assert cfg['num_sweep_frames'] == 1 and cfg['binomial_p'] == 0
    assert not cfg['with_history_frames'] and not cfg['fusion']['args']['proj_first']
    assert not cfg['noise_setting']['add_noise'] and not cfg['is_generate_gt_flow']
    cfg['preprocess']['voxel_backend'] = 'spconv2'
    root = Path(cfg['data_dir'])
    vehicle = sensor_index(root, 'vehicle-side')
    infrastructure = sensor_index(root, 'infrastructure-side')
    report['protocol'] = {'data_root': str(root), 'sample_count_per_split': args.samples,
        'seed': args.seed, 'reader': cfg['fusion']['core_method'], 'added_delay_frames': 0,
        'history_frames': 0, 'preprocess_backend': 'spconv2',
        'reader_voxel_size': cfg['preprocess']['args']['voxel_size'],
        'probe_voxel_size': args.voxel_size, 'lidar_range': cfg['preprocess']['cav_lidar_range'],
        'augmentation': False, 'probe_batchnorm': 'train', 'probe_weights': 'random',
        'timestamp_note': 'Reader timestamp is a frame ID; sensor timestamps below are raw metadata units.'}
    report['splits'] = {}
    for split, split_path in [('train', cfg['root_dir']), ('val', cfg['validate_dir'])]:
        local_cfg = copy.deepcopy(cfg)
        local_cfg['validate_dir'] = split_path
        # Both splits are audited without training augmentation.
        dataset = build_dataset(local_cfg, visualize=False, train=False)
        if len(dataset) < args.samples:
            raise ValueError('Not enough indexed data in {}'.format(split))
        selected = np.linspace(0, len(dataset) - 1, args.samples, dtype=int).tolist()
        result = {'split_file': split_path, 'split_sha256': file_hash(split_path),
                  'split_entries': len(dataset.data_split), 'reader_entries': len(dataset), 'samples': []}
        report['splits'][split] = result
        for index in selected:
            sample = {'index': index, 'vehicle_frame_id': dataset.data[index]}
            result['samples'].append(sample)
            base = dataset.retrieve_base_data(index)
            poses = [base[i]['curr']['params']['lidar_pose'] for i in (0, 1)]
            to_ego = np.asarray(x1_to_x2(poses[1], poses[0]))
            inverse = np.asarray(x1_to_x2(poses[0], poses[1]))
            assert np.isfinite(to_ego).all()
            assert np.allclose(to_ego @ inverse, np.eye(4), atol=1e-5)
            assert np.allclose(x1_to_x2(poses[0], poses[0]), np.eye(4), atol=1e-5)
            veh_id, inf_id = [base[i]['curr']['frame_id'] for i in (0, 1)]
            veh_time = vehicle[veh_id].get('pointcloud_timestamp')
            inf_time = infrastructure[inf_id].get('pointcloud_timestamp')
            sample.update({'infrastructure_frame_id': inf_id, 'infra_to_ego': to_ego.tolist(),
                'vehicle_pointcloud_timestamp_raw': veh_time, 'infra_pointcloud_timestamp_raw': inf_time,
                'infra_minus_vehicle_timestamp_raw': None if veh_time is None or inf_time is None
                                                     else int(inf_time) - int(veh_time)})
            assert base[1]['past_k'][0]['frame_id'] == inf_id
            processed = dataset[index]
            if processed is None:
                raise ValueError('Reader returned None; no silent sample replacement')
            batch = dataset.collate_batch_test([processed])
            assert batch is not None and batch['ego']['record_len'].tolist() == [2]
            sample['gt_boxes_in_range'] = int(processed['ego']['object_bbx_mask'].sum())
            sample['collated_voxel_shape'] = list(batch['ego']['processed_lidar']['voxel_features'].shape)
            sample['sparse_probe'] = sparse_probe([base[i]['curr']['lidar_np'] for i in (0, 1)],
                                                 cfg['preprocess']['cav_lidar_range'], args.voxel_size)
            sample['status'] = 'passed'
            print('{} index={} vehicle={} passed'.format(split, index, veh_id), flush=True)
    report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='opencood/hypes_yaml/dair-v2x/repro/dair_cobevflow_stage1_sync_where2comm_wide.yaml')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--voxel-size', type=float, nargs=3, default=[0.4, 0.4, 0.1])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.samples < 1 or min(args.voxel_size) <= 0:
        parser.error('samples and voxel sizes must be positive')
    report = {'kind': 'cp_snn_readiness_only', 'status': 'running',
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
              'git_status': subprocess.check_output(['git', 'status', '--porcelain']).decode(),
              'config': args.config, 'config_sha256': file_hash(args.config), 'arguments': vars(args)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args, report)
        report['status'] = 'passed'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
