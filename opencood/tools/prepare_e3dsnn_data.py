"""Audit every vehicle PCD before training; record explicit usable splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import pcd_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    cfg = load_yaml('opencood/hypes_yaml/dair-v2x/snn/e3dsnn_vehicle_single.yaml')
    root = Path(cfg['data_dir'])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous audits; choose a fresh output directory')
    output.mkdir(parents=True)
    info_path = root / 'cooperative/data_info.json'
    info = {Path(row['vehicle_pointcloud_path']).stem: row for row in json.loads(info_path.read_text())}
    report = {'data_root': str(root), 'cooperative_info_sha256': hashlib.sha256(info_path.read_bytes()).hexdigest(),
              'lidar_range': cfg['preprocess']['cav_lidar_range'], 'splits': {}}

    def inspect(frame_id):
        try:
            path = root / info[frame_id]['vehicle_pointcloud_path']
            points, _ = pcd_utils.read_pcd(str(path))
            raw = len(points)
            if raw == 0:
                return {'frame_id': frame_id, 'valid': False, 'reason': 'empty_pcd', 'raw_points': 0}
            if not np.isfinite(points).all():
                return {'frame_id': frame_id, 'valid': False, 'reason': 'nonfinite_points'}
            points = pcd_utils.mask_ego_points(pcd_utils.mask_points_by_range(points, report['lidar_range']))
            return {'frame_id': frame_id, 'valid': len(points) > 0,
                    'reason': 'ok' if len(points) else 'empty_after_range_and_ego_mask',
                    'raw_points': raw, 'usable_points': len(points), 'file_size': path.stat().st_size}
        except Exception as exc:
            return {'frame_id': frame_id, 'valid': False, 'reason': '{}: {}'.format(type(exc).__name__, exc)}

    for name, split_path in [('train', cfg['root_dir']), ('val', cfg['validate_dir'])]:
        ids = json.loads(Path(split_path).read_text())
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(inspect, ids))
        valid = [row['frame_id'] for row in rows if row['valid']]
        report['splits'][name] = {'source_path': split_path,
            'source_sha256': hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),
            'original_count': len(ids), 'valid_count': len(valid), 'valid_ids': valid,
            'excluded': [row for row in rows if not row['valid']]}
        (output / (name + '_point_audit.json')).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        (output / (name + '.json')).write_text(json.dumps(valid, indent=2), encoding='utf-8')
        print('{}: {}/{} usable; {} excluded'.format(name, len(valid), len(ids), len(ids)-len(valid)), flush=True)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
