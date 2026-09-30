"""Freeze a paired DAIR split after checking both clouds, poses and labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.paired_dair_fusion import load_pair_geometry
from opencood.utils import pcd_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    root, out = Path(cfg['data_dir']), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    info_path = root / 'cooperative/data_info.json'
    info = {Path(r['vehicle_pointcloud_path']).stem: r for r in json.loads(info_path.read_text())}
    times = []
    for side in ('vehicle-side', 'infrastructure-side'):
        rows = json.loads((root / side / 'data_info.json').read_text())
        times.append({Path(r['pointcloud_path']).stem: int(r['pointcloud_timestamp']) for r in rows})
    region = cfg['preprocess']['cav_lidar_range']

    def inspect(fid):
        try:
            row = info[fid]
            iid = Path(row['infrastructure_pointcloud_path']).stem
            _, matrix, shift = load_pair_geometry(root, row)
            clouds = [pcd_utils.read_pcd(str(root / row[k]))[0] for k in
                      ('vehicle_pointcloud_path', 'infrastructure_pointcloud_path')]
            if any(len(p) == 0 or not np.isfinite(p).all() for p in clouds):
                raise ValueError('Empty/nonfinite raw paired cloud')
            raw_road_count = len(pcd_utils.mask_points_by_range(clouds[1], region))
            clouds[1][:, 2] += shift
            usable = [len(pcd_utils.mask_ego_points(pcd_utils.mask_points_by_range(clouds[0], region))),
                      len(pcd_utils.mask_points_by_range(clouds[1], region))]
            if not all(usable):
                raise ValueError('Empty paired cloud after range/height normalization')
            labels = json.loads((root / row['cooperative_label_path']).read_text())
            if any(np.asarray(o['world_8_points']).shape != (8, 3) or
                   not np.isfinite(o['world_8_points']).all() for o in labels):
                raise ValueError('Invalid cooperative world box')
            timestamp = [times[0][fid], times[1][iid]]
            return {'frame_id': fid, 'infra_id': iid, 'valid': True, 'raw_points': [len(p) for p in clouds],
                    'usable_points': usable, 'road_raw_crop': raw_road_count, 'road_z_shift': shift,
                    'timestamps_us': timestamp, 'delta_us': timestamp[1] - timestamp[0],
                    'road_to_ego': matrix.tolist(), 'labels': len(labels)}
        except Exception as exc:
            return {'frame_id': fid, 'valid': False, 'reason': '{}: {}'.format(type(exc).__name__, exc)}

    report = {'protocol': 'paired_dair_fusion_v1', 'data_root': str(root), 'lidar_range': region,
        'road_height_policy': 'source-local z shift T[2,3]/T[2,2]; adjusted full SE3; recipient-dependent, calibration-only',
        'cooperative_info_sha256': hashlib.sha256(info_path.read_bytes()).hexdigest(),
        'timestamps_us': {}, 'splits': {}}
    for split, key in [('train', 'root_dir'), ('val', 'validate_dir')]:
        source = Path(cfg[key]); ids = json.loads(source.read_text())
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate frame IDs')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(inspect, ids))
        (out / (split + '_audit.json')).write_text(json.dumps(rows, indent=2))
        valid = [r['frame_id'] for r in rows if r['valid']]
        if not valid:
            raise RuntimeError('No valid paired frames; inspect data contract')
        report['timestamps_us'].update({r['frame_id']: r['timestamps_us'] for r in rows if r['valid']})
        report['splits'][split] = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'original_count': len(ids), 'valid_count': len(valid), 'valid_ids': valid,
            'excluded': [r for r in rows if not r['valid']]}
        print(json.dumps({'split': split, 'valid': len(valid), 'original': len(ids)}), flush=True)
    if set(report['splits']['train']['valid_ids']) & set(report['splits']['val']['valid_ids']):
        raise ValueError('Train/validation overlap')
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
