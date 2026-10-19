"""Bounded CPU loss decomposition on saved reset-policy checkpoints.

This cannot reproduce a mid-epoch optimizer state or the original voxel order.
It diagnoses the named training samples without changing training or labels.
"""
import argparse
import json
from pathlib import Path

import torch

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_unified_bev import UnifiedBEVDetector
from opencood.tools.launch_int_memory import digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json


def run(a, out):
    for key, value in dict(mode='gru', stage='pilot', device='cpu',
                           content_calibration=None, periodic_last=0).items():
        setattr(a, key, value)
    teacher, cfg, _, provenance = setup(a)
    runtime = json.loads((Path(a.root) / 'full_gru/effective_runtime.json').read_text())['arguments']
    requested = set(a.frames)
    assert len(requested) == len(a.frames)
    rows = [row for row in json.loads(Path(runtime['manifest']).read_text())['train']
            if row['frame'] in requested]
    assert len(rows) == len(requested) and all(row['supervised'] for row in rows)
    reports = []
    for checkpoint in a.checkpoints:
        source = Path(checkpoint)
        sha = digest(source)
        assert json.loads((source.parent / 'report.json').read_text())['checkpoint_sha256'] == sha
        saved = torch.load(str(source), map_location='cpu')
        contract = saved['contract']
        assert contract['memory_policy'] == 'reset'
        assert contract['data_contract'] == provenance['contract']
        variant = contract['version'].replace('unified_bev_dense_', '')
        assert variant in ('v2', 'v2n')
        model = UnifiedBEVDetector(teacher, cfg, variant, 'reset').cpu().train()
        model.load_state_dict(saved['model'], strict=True)
        criterion = PointPillarDirLoss(cfg['loss']['args']).cpu()
        _, stream = loader(rows, cfg, 0, a.seed)
        records = []
        with torch.no_grad():
            for row, sample in zip(rows, stream):
                assert row['frame'] == sample['meta']['frame']
                data = to_device(sample['ego'], torch.device('cpu'))
                prediction, _, _ = model.step(data, None, sample['meta'])
                value = criterion(prediction, data['label_dict'])
                assert bool(torch.isfinite(value))
                records.append(dict(frame=row['frame'], segment=row['segment'],
                    loss=float(value), components={k: float(v) for k, v in criterion.loss_dict.items()},
                    positives=int(data['label_dict']['pos_equal_one'].sum()),
                    target_absmax=float(data['label_dict']['targets'].abs().max()),
                    voxels=int(data['processed_lidar']['voxel_features'].shape[0]),
                    prediction_ranges={k: [float(v.min()), float(v.max())] for k, v in prediction.items()
                                       if k in ('psm', 'rm', 'dm')}))
        assert len(records) == len(rows)
        reports.append(dict(checkpoint=str(source), sha256=sha, epoch=saved['epoch'],
                            variant=variant, records=records))
    write_json(out / 'report.json', dict(status='COMPLETED', device='cpu', split='train',
        seed=a.seed, reports=reports, caveat='Saved epoch-end weights, CPU and subset loader; not exact '
        'mid-epoch weights/RNG/voxel order or optimizer replay. No training/data changes; no AP/timing claim.'))
    write_json(out / 'status.json', dict(status='COMPLETED'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--frames', nargs='+', required=True)
    p.add_argument('--seed', type=int, default=303)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'status.json', dict(status='RUNNING'))
    try:
        run(a, out)
    except Exception as exc:
        write_json(out / 'status.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
