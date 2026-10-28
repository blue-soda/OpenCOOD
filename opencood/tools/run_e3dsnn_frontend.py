"""One matched ordinary/deeper SNN frontend experiment, fixed IF byte budget."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.e3dsnn_snapshot_frontend import E3dsnnSnapshotFrontend
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import checked_forward, write_json
from opencood.tools.run_e3dsnn_capacity import SOURCE_SHA, SEED, BUDGETS, REPO, sha, read
from opencood.utils.task_snapshot_packet import HEADER, decode

ARMS = ('f0_base', 'f1_deep')


def prepare(root, config, checkpoint):
    assert sha(checkpoint) == SOURCE_SHA
    root.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(Path(config).read_text())
    cfg['model']['core_method'] = 'e3dsnn_snapshot_frontend'
    args = cfg['model']['args']
    args.pop('capacity', None)
    args['snapshot_joint'].update(train_budgets=[2304], ego_aux_weight=.5)
    args['snapshot'].update(budget_bytes=2304, packet_directory=None)
    args['message_dropout'] = .1
    cfg['train_params'].update(batch_size=2, epoches=10)
    cfg['optimizer']['lr'] = 5e-5
    cfg['lr_scheduler'].update(step_size=[7, 9], gamma=.1)
    source = torch.load(checkpoint, map_location='cpu')
    hashes, counts = {}, {}
    for arm, depth in zip(ARMS, (0, 2)):
        folder = root / arm
        folder.mkdir()
        args['frontend'] = dict(extra_blocks_per_stage=depth)
        model = E3dsnnSnapshotFrontend(args)
        missing, extra = model.load_state_dict(source, strict=False)
        new = {k for k in model.state_dict() if k not in source}
        # Torch 1.10 BN loading synthesizes legacy num_batches_tracked buffers
        # and omits them from missing_keys; no learned tensor may be omitted.
        auto_counters = {k for k in new if k.endswith('.num_batches_tracked')}
        assert set(missing) == new - auto_counters and not extra
        assert all(model.state_dict()[k].item() == 0 for k in auto_counters)
        assert all(k.startswith(tuple('backbone_3d.conv%d.blocks.%d.' % (s, i)
                         for s in (3, 4) for i in (2, 3))) for k in new)
        assert bool(new) == bool(depth)
        for k, value in source.items():
            assert torch.equal(value, model.state_dict()[k]), k
        torch.save(model.state_dict(), str(folder / 'initial.pth'))
        counts[arm] = dict(total=sum(p.numel() for p in model.parameters()),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            additional_state_keys=sorted(new))
        for label, budget in [('train', 2304)] + [('b' + str(b), b) for b in BUDGETS]:
            c = copy.deepcopy(cfg)
            c['model']['args']['snapshot']['budget_bytes'] = budget
            if label != 'train':
                c['model']['args']['snapshot']['packet_directory'] = str(folder / label / 'packets')
            path = folder / (label + '.yaml')
            path.write_text(yaml.safe_dump(c))
            hashes[str(path.relative_to(root))] = sha(path)
        hashes[str((folder / 'initial.pth').relative_to(root))] = sha(folder / 'initial.pth')
    write_json(root / 'protocol.json', dict(source_sha256=SOURCE_SHA, source=str(checkpoint),
        source_config_sha256=sha(config), hashes=hashes, counts=counts, seed=SEED,
        dataset_hashes={k: sha(cfg[k]) for k in ('root_dir', 'validate_dir', 'data_manifest')},
        primary_budget=2304, final_budgets=BUDGETS, epochs=10,
        loss='Ldet(2304)+0.5 Ldet(0); dropout .1; all 3D trainable, BN statistics frozen',
        numerical_policy={'cuda_matmul_tf32': False, 'cudnn_tf32': False},
        selection='full 2304B AP70 including epoch0; common CPU GT final replay',
        training_timeout_seconds=345600, evaluation_timeout_seconds=28800,
        gpu_memory_fraction=.25, initialization='identity residual branches via zero final BN affine',
        no_hard_parameter_ceiling=True))
    print(json.dumps({k: {n: v[n] for n in ('total', 'trainable')} for k, v in counts.items()}))


def packet_content(packet):
    if not packet:
        return b'', None
    meta_len = HEADER.unpack(packet[:HEADER.size])[1]
    meta = json.loads(packet[HEADER.size:HEADER.size + meta_len])
    counts, scales = decode(packet, meta)
    return packet[HEADER.size + meta_len:], (counts, scales)


def audit(root, arm):
    from opencood.data_utils.datasets import build_dataset
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_per_process_memory_fraction(.25); torch.set_num_threads(1)
    folder = root / arm
    cfg = load_yaml(str(folder / 'train.yaml'))
    data, train = build_dataset(cfg, train=False), build_dataset(cfg, train=True)
    assert len(data) == 1738 and len(train) == 4678
    model = train_utils.create_model(cfg).cuda()
    model.load_state_dict(torch.load(str(folder / 'initial.pth'), map_location='cpu'), strict=True)
    ref_cfg = load_yaml(str(root / ARMS[0] / 'train.yaml'))
    reference = train_utils.create_model(ref_cfg).cuda()
    reference.load_state_dict(torch.load(str(root / ARMS[0] / 'initial.pth'), map_location='cpu'), strict=True)
    model.eval(); reference.eval()
    records = []
    with torch.no_grad():
        for index in (0, 579, 1158, 1737):
            batch = train_utils.to_device(data.collate_batch_test([data[index]]), torch.device('cuda'))
            actual, original = model(batch['ego']), reference(batch['ego'])
            diffs = {k: float((actual[k] - original[k]).abs().max()) for k in ('psm', 'rm', 'dm')}
            assert all(torch.allclose(actual[k], original[k], atol=1e-4, rtol=2e-5) for k in diffs), diffs
            model.packet_roundtrip = reference.packet_roundtrip = True
            packets, ref_packets = [], []
            model._snapshot_packet_observer = lambda fid, blob: packets.append(blob)
            reference._snapshot_packet_observer = lambda fid, blob: ref_packets.append(blob)
            wire = model(batch['ego']); reference(batch['ego'])
            assert len(packets) == len(ref_packets) == 1
            assert len(packets[0]) == len(ref_packets[0]) <= 2304
            body, values = packet_content(packets[0]); ref_body, ref_values = packet_content(ref_packets[0])
            assert body == ref_body
            if values is not None:
                assert all(np.array_equal(a, b) for a, b in zip(values, ref_values))
            assert all(torch.equal(actual[k], wire[k]) for k in diffs)
            records.append(dict(index=index, bytes=len(packets[0]), max_abs=diffs,
                decoded_content_equal=True, compressed_payload_equal=True, direct_wire_exact=True,
                packet_sha256=hashlib.sha256(packets[0]).hexdigest(),
                reference_packet_sha256=hashlib.sha256(ref_packets[0]).hexdigest()))
            model.packet_roundtrip = reference.packet_roundtrip = False
            model._snapshot_packet_observer = reference._snapshot_packet_observer = None
        model.snapshot_budget = 0
        zero = model(batch['ego']); model.fusion_mode = 'ego'; ego = model(batch['ego'])
        model.fusion_mode = 'residual'; model.snapshot_budget = 2304
        assert all(torch.equal(zero[k], ego[k]) for k in ('psm', 'rm', 'dm'))
    del reference, batch, original, actual, wire, zero, ego
    torch.cuda.empty_cache(); model.train()
    fixed = model.fixed_state_digest()
    batch = train_utils.to_device(train.collate_batch_train([train[0], train[1]]), torch.device('cuda'))
    criterion = train_utils.create_loss(cfg).cuda()
    optimizer = train_utils.setup_optimizer(cfg, model)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    included = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert expected.issubset(set(included)) and len(included) == len(set(included))
    prefixes = ['backbone_3d.conv_input.', 'backbone_3d.conv1.', 'backbone_3d.conv2.',
        'backbone_3d.conv3.', 'backbone_3d.conv4.', 'backbone_3d.conv_out.',
        'content.project.', 'content.expand.', 'joint_receiver.',
        'backbone_2d.', 'cls_head.', 'reg_head.', 'dir_head.']
    if arm == 'f1_deep':
        prefixes += ['backbone_3d.conv%d.blocks.%d.' % (s, i) for s in (3, 4) for i in (2, 3)]
    gradient_steps = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, loss = checked_forward(model, criterion, batch)
        loss.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        grads = {prefix: sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                           if n.startswith(prefix) and p.grad is not None) for prefix in prefixes}
        assert all(np.isfinite(x) and x > 0 for x in grads.values()), grads
        if step == 1 and arm == 'f1_deep':
            for prefix in prefixes[-4:]:
                conv_grad = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                    if n.startswith(prefix) and p.ndim == 5 and p.grad is not None)
                assert np.isfinite(conv_grad) and conv_grad > 0, prefix
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        assert model.fixed_state_digest() == fixed
        gradient_steps.append(dict(loss=float(loss.detach()), gradients=grads))
    write_json(folder / 'audit.json', dict(passed=True, records=records, gradient_steps=gradient_steps,
        zero_equals_ego=True, frozen_state_unchanged=True, optimizer_coverage=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), all_audit_updates_discarded=True))


def coordinate(root, arm, common_gt):
    folder, protocol = root / arm, read(root / 'protocol.json')
    for name, digest in protocol['hashes'].items():
        assert sha(root / name) == digest
    cfg = load_yaml(str(folder / 'train.yaml'))
    for key, digest in protocol['dataset_hashes'].items():
        assert sha(cfg[key]) == digest
    def run(stage, command, seconds):
        write_json(folder / 'status.json', dict(state='running', stage=stage, unix=time.time()))
        with (folder / (stage + '.log')).open('xb') as log:
            subprocess.run(['timeout', '--kill-after=30', str(seconds)] + command,
                cwd=str(REPO), env=os.environ.copy(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, check=True, start_new_session=True)
    try:
        run('audit', [sys.executable, '-u', '-m', __spec__.name, '--mode', 'audit',
            '--root', str(root), '--arm', arm], 1800)
        assert read(folder / 'audit.json')['passed']
        base = [sys.executable, '-u', '-m', 'opencood.tools.train_e3dsnn_single',
                '--workers', '2', '--seed', str(SEED)]
        run('train', base + ['--mode', 'train', '--config', str(folder / 'train.yaml'),
            '--checkpoint', str(folder / 'initial.pth'), '--output', str(folder / 'train'),
            '--epochs', '10', '--selection-iou', '0.7', '--retain-initial-best'],
            protocol['training_timeout_seconds'])
        summary = {}
        for budget in BUDGETS:
            name = 'b' + str(budget); out = folder / name
            run(name, base + ['--mode', 'eval', '--config', str(folder / (name + '.yaml')),
                '--checkpoint', str(folder / 'train/best_ap70.pth'), '--output', str(out),
                '--packet-roundtrip', '--eval-3d', '--save-predictions'], protocol['evaluation_timeout_seconds'])
            run(name + '_replay', [sys.executable, '-m', 'opencood.tools.replay_e3dsnn_predictions',
                '--evaluation-dir', str(out)], 3600)
            run(name + '_common', [sys.executable, '-m', 'opencood.tools.replay_capacity_common_gt',
                '--evaluation-dir', str(out), '--reference', str(common_gt), '--budget', str(budget)], 3600)
            summary[name] = read(out / 'common_gt_evaluation.json')
        write_json(folder / 'summary_common_gt.json', summary)
        write_json(folder / 'status.json', dict(state='complete'))
    except Exception:
        write_json(folder / 'status.json', dict(state='failed', error=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'audit', 'coordinate'), required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--source-config'); parser.add_argument('--source-checkpoint')
    parser.add_argument('--common-gt-reference')
    a = parser.parse_args(); root = Path(a.root)
    if a.mode == 'prepare':
        prepare(root, a.source_config, a.source_checkpoint)
    elif a.mode == 'audit':
        audit(root, a.arm)
    else:
        coordinate(root, a.arm, a.common_gt_reference)
