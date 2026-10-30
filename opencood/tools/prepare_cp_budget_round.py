"""Register F0 budget adaptation and recover R-020 evaluation in fresh folders."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import yaml

from opencood.tools.run_registered_research import sha

F0_SHA = 'be320dc564bf98d66e76e8f034e0d50647a01888459f4c76665a88a05c1e3841'
REPO = Path(__file__).resolve().parents[2]


def dump(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2)


def job(name, module, args, seconds, outputs=()):
    return dict(name=name, command=['{python}', '-u', '-m', module] + list(map(str, args)),
                timeout_seconds=seconds, required_outputs=list(map(str, outputs)))


def eval_jobs(root, budget, config, checkpoint, reference):
    out = root / ('b' + str(budget))
    return [job(out.name, 'opencood.tools.train_e3dsnn_single',
        ['--mode', 'eval', '--config', config, '--checkpoint', checkpoint,
         '--output', out, '--workers', 2, '--seed', 20260929,
         '--packet-roundtrip', '--eval-3d', '--save-predictions'], 86400,
        [out / 'evaluation.json']),
        job(out.name + '_native', 'opencood.tools.replay_e3dsnn_predictions',
            ['--evaluation-dir', out], 7200, [out / 'offline_replay.json']),
        job(out.name + '_common', 'opencood.tools.replay_capacity_common_gt',
            ['--evaluation-dir', out, '--reference', reference, '--budget', budget],
            7200, [out / 'common_gt_evaluation.json'])]


def prepare(a):
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(Path(a.config).read_text())
    source = Path(a.checkpoint)
    frozen = root / 'initial.pth'
    shutil.copyfile(str(source), str(frozen))
    assert sha(frozen) == sha(source)
    if a.kind == 'adapt':
        assert sha(frozen) == F0_SHA
        assert cfg['model']['core_method'] == 'e3dsnn_snapshot_frontend'
        cfg['data_dir'] = a.data_root
        for key, file in [('root_dir', 'train.json'), ('validate_dir', 'val.json'), ('test_dir', 'val.json')]:
            cfg[key] = str(Path(a.data_root) / file)
        cfg['data_manifest'] = a.manifest
        cfg['model']['args']['snapshot_joint']['train_budgets'] = [a.budget]
        cfg['model']['args']['snapshot']['budget_bytes'] = a.budget
        cfg['model']['args']['snapshot']['packet_directory'] = None
        cfg['train_params'].update(epoches=6, batch_size=2)
        cfg['optimizer']['lr'] = 2e-5
        cfg['lr_scheduler'].update(step_size=[4, 5], gamma=.1)
        train_config = root / 'train.yaml'
        train_config.write_text(yaml.safe_dump(cfg))
        jobs = [job('smoke', 'opencood.tools.audit_cp_frontend_budget',
            ['--config', train_config, '--checkpoint', frozen,
             '--output', root / 'audit.json'], 3600, [root / 'audit.json']),
            job('train', 'opencood.tools.train_e3dsnn_single',
            ['--mode', 'train', '--config', train_config, '--checkpoint', frozen,
             '--output', root / 'train', '--epochs', 6, '--workers', 2,
             '--seed', 20260929, '--selection-iou', '0.7', '--retain-initial-best'],
            345600, [root / 'train/status.json', root / 'train/best_ap70.pth'])]
        checkpoint = root / 'train/best_ap70.pth'
        budgets = [a.budget, 0, 4096 if a.budget == 2304 else 2304]
    else:
        old = Path(a.old_root)
        assert json.loads((old / 'train/status.json').read_text())['state'] == 'complete'
        metrics = [json.loads(s) for s in (old / 'train/metrics.jsonl').read_text().splitlines()]
        assert metrics[-1]['epoch'] == 10
        jobs = []
        for budget in (0, 2048, 4096):
            dest = root / ('b' + str(budget))
            dest.mkdir()
            for name in ('evaluation.json', 'offline_replay.json'):
                shutil.copyfile(str(old / dest.name / name), str(dest / name))
            for name in ('predictions', 'packets'):
                shutil.copytree(str(old / dest.name / name), str(dest / name), copy_function=os.link)
            jobs.append(job(dest.name + '_common', 'opencood.tools.replay_capacity_common_gt',
                ['--evaluation-dir', dest, '--reference', a.reference, '--budget', budget],
                7200, [dest / 'common_gt_evaluation.json']))
        checkpoint, budgets = frozen, [8192]
    for budget in budgets:
        c = copy.deepcopy(cfg)
        c['model']['args']['snapshot'].update(budget_bytes=budget,
            packet_directory=str(root / ('b' + str(budget)) / 'packets'))
        path = root / ('b' + str(budget) + '.yaml')
        path.write_text(yaml.safe_dump(c))
        jobs += eval_jobs(root, budget, path, checkpoint, a.reference)
    inputs = [frozen, Path(a.config)] + list(root.glob('*.yaml'))
    inputs += [Path(cfg[k]) for k in ('root_dir', 'validate_dir', 'data_manifest')]
    inputs += [Path(a.reference) / 'common_gt_manifest.json', Path(a.reference) / 'common_val.json']
    protocol = dict(repo=str(REPO), code_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
        input_hashes={str(p): sha(p) for p in inputs}, jobs=jobs,
        kind=a.kind, seed=20260929, source=str(source), source_sha256=sha(source),
        primary_budget=a.budget if a.kind == 'adapt' else 2048,
        hypothesis='Test whether budget-matched full-frontend adaptation improves AP70',
        selection='native full validation AP70 at registered primary budget; include epoch0; fixed common GT final replay',
        training='adapt: six epochs, same F0 source, lr2e-5, decay4/5, Ldet+.5Lego, dropout .1, frozen BN statistics',
        exit_rule='No extensions or post-hoc budget selection; report every arm, epoch0 and all final workpoints',
        promotion='Exploratory single seed only; repeat and matched mechanism controls required before paper claim',
        recovery='Fresh outputs; old R020 predictions/packets hardlinked read-only, source best copied; no retraining',
        scope='Preserve IF8, local membrane, Count4 and ANN BEV; no architectural novelty claimed')
    dump(root / 'protocol.json', protocol)
    print(json.dumps(dict(root=str(root), jobs=len(jobs), source_sha256=sha(source))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('adapt', 'recover'), required=True)
    for key in ('root', 'config', 'checkpoint', 'reference'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--budget', type=int, choices=(2304, 4096))
    p.add_argument('--data-root'); p.add_argument('--manifest'); p.add_argument('--old-root')
    a = p.parse_args()
    if a.kind == 'adapt' and not all((a.budget, a.data_root, a.manifest)):
        p.error('adapt requires budget, data-root and manifest')
    if a.kind == 'recover' and not a.old_root:
        p.error('recover requires old-root')
    prepare(a)
