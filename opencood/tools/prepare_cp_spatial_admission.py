"""Register real-data admission only; no training/evaluation queue is launched."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import torch
import yaml
from opencood.models.fuse_modules.e3dsnn_spatial_drive import SnapshotSpatialDrive
from opencood.tools.prepare_cp_budget_round import F0_SHA, REPO, dump, job
from opencood.tools.run_registered_research import sha


def prepare(a):
    source = Path(a.checkpoint)
    assert sha(source) == F0_SHA
    cfg = yaml.safe_load(Path(a.config).read_text())
    assert cfg['model']['core_method'] == 'e3dsnn_snapshot_frontend'
    manifest = json.loads(Path(a.manifest).read_text())
    assert manifest['data_root'] == a.data_root
    assert [manifest['splits'][s]['valid_count'] for s in ('train', 'val')] == [4678, 1738]
    root = Path(a.root); root.mkdir(parents=True, exist_ok=False)
    cfg['data_dir'] = a.data_root
    for key, file in [('root_dir', 'train.json'), ('validate_dir', 'val.json'), ('test_dir', 'val.json')]:
        cfg[key] = str(Path(a.data_root) / file)
    cfg['data_manifest'] = a.manifest
    cfg['model']['core_method'] = 'e3dsnn_snapshot_spatial'
    cfg['model']['args']['snapshot_joint']['train_budgets'] = [2304]
    cfg['model']['args']['snapshot'].update(budget_bytes=2304, packet_directory=None)
    cfg['optimizer']['lr'] = 2e-5
    jobs = []; inputs = [source, Path(a.config), Path(a.manifest)]
    inputs += [Path(cfg[k]) for k in ('root_dir', 'validate_dir')]
    initial_hashes = []
    for condition in ('spatial', 'global'):
        folder = root / condition; folder.mkdir()
        c = copy.deepcopy(cfg)
        c['model']['args']['snapshot_spatial'] = dict(condition=condition)
        state = torch.load(str(source), map_location='cpu')
        gate = SnapshotSpatialDrive(condition)
        additions = {'snapshot_spatial_drive.'+k:v for k,v in gate.state_dict().items()}
        assert not set(state).intersection(additions)
        state.update(additions)
        checkpoint = folder / 'initial.pth'; torch.save(state, str(checkpoint))
        config = folder / 'audit.yaml'; config.write_text(yaml.safe_dump(c))
        inputs += [checkpoint, config]; initial_hashes.append(sha(checkpoint))
        jobs.append(job(condition, 'opencood.tools.audit_cp_spatial',
            ['--config', config, '--checkpoint', checkpoint, '--parent', source,
             '--output', folder / 'audit.json'], 3600, [folder / 'audit.json']))
    assert initial_hashes[0] == initial_hashes[1]
    for folder in a.dependency_paths:
        inputs += [p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    inputs += list((REPO / 'opencood/utils').glob('box_overlaps*.so'))
    protocol = dict(repo=str(REPO), code_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
        input_hashes={str(p):sha(p) for p in inputs}, jobs=jobs,
        dependency_paths=a.dependency_paths, source_sha256=F0_SHA,
        initial_sha256=initial_hashes[0], seed=20260929, budget=2304,
        scope='Admission only: two val frames, two train frames, two discarded optimizer steps per arm',
        hypothesis='Identity, actual wire counts and geometry, 0B, gradient coverage and frozen-state checks',
        boundary='No full validation or training experiment; no AP or mechanism conclusion',
        exit_rule='Stop after the two audits; any failure exits without automatic retry')
    dump(root / 'protocol.json', protocol)
    print(json.dumps(dict(root=str(root), jobs=len(jobs), initial_sha256=initial_hashes[0])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root', 'config', 'checkpoint', 'data-root', 'manifest'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--dependency-paths', nargs='*', default=[])
    prepare(p.parse_args())
