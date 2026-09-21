"""Audit the reader transform against independently reconstructed real sensor poses."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import torch
import yaml
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd())
from opencood.tools.sfr_run import seed, plain, digest
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.sfr_dair import SfrDAIRDataset
from opencood.models.sfr.geometry import reader_to_reference, transform_points
from opencood.utils.transformation_utils import x1_to_x2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--split_file', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max_steps', type=int, default=32)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    base = load_yaml(cfg['base_config'])
    base['validate_dir'] = args.split_file
    base['binomial_p'] = .3
    base['ectra_ego_history'] = dict(enabled=True, max_age_ms=100)
    seed(303)
    data = SfrDAIRDataset(copy.deepcopy(base), False, False)
    assert not data.proj_first, 'This SFR interface requires native sender coordinates'
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=data.collate_batch_test, generator=torch.Generator().manual_seed(303))
    records = []
    probes = torch.tensor([[0., 0., 0.], [10., 5., 1.], [-15., -3., -1.]], dtype=torch.float64)
    for index, batch in enumerate(loader):
        if index >= args.max_steps: break
        if batch is None: continue
        ego = batch['ego']; meta = ego['sfr_metadata'][0]
        reference = meta['observations'][0][0]['nominal_pose']
        for agent, frames in enumerate(meta['observations']):
            for frame, item in enumerate(frames):
                expected = torch.tensor(x1_to_x2(item['nominal_pose'], reference), dtype=torch.float64)
                reader = ego['pairwise_t_matrix'][0, agent, frame].double()
                corrected = reader_to_reference(reader)
                matrix_error = float((corrected-expected).abs().max())
                point_error = float((transform_points(probes, corrected)-transform_points(probes, expected)).abs().max())
                legacy_error = float((reader-expected).abs().max())
                if matrix_error > 2e-4 or point_error > 1e-3:
                    raise AssertionError((index, agent, frame, matrix_error, point_error))
                records.append(dict(index=index, agent=agent, frame=item['frame_id'],
                                    corrected_matrix_max_error=matrix_error, corrected_point_max_error_m=point_error,
                                    legacy_matrix_max_error=legacy_error))
    assert records and max(r['legacy_matrix_max_error'] for r in records) > 1.
    result = dict(split_sha256=digest(args.split_file), records=records, passed=True,
                  max_matrix_error=max(r['corrected_matrix_max_error'] for r in records),
                  max_point_error_m=max(r['corrected_point_max_error_m'] for r in records),
                  note='Real metadata poses reconstructed independently with x1_to_x2; noise-free nominal audit')
    Path(args.output).write_text(json.dumps(plain(result), indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}))


if __name__ == '__main__': main()
