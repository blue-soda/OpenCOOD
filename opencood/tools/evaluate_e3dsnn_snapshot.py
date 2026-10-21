"""Bind actual cumulative-snapshot bytes to decoded full-split AP, without training.

Budgets name the ORIGINAL session_zlib event-prefix selection, not a snapshot
budget. One snapshot replaces source counts; it is never added repeatedly.
Internal event order is discarded. This is not an online incremental protocol.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, evaluation
from opencood.tools.train_e3dsnn_single import validate, write_json
from opencood.tools.profile_e3dsnn_event_granularity import encode_snapshot, decode_snapshot

SOURCE_SHA = '60b7e0ff84e992862902aa9325b56ac13ab5658d70c4aacb194028638f5105dc'
BUDGETS = (4096, 8192, 16384)
SEMANTICS = ('Original session_zlib/256-event-packet prefix reference budget; '
    'actual transport is one E3GS1 cumulative count snapshot with full metadata. '
    'No event-order preservation, no snapshot-budget optimization, no online incremental protocol.')


class SnapshotEvaluator(torch.nn.Module):
    packet_roundtrip = True

    def __init__(self, model, reference_budget, folder):
        super().__init__()
        from opencood.utils.e3dsnn_transplant import encoder_digest
        self.model = model
        self.budget = reference_budget
        self.folder = folder
        self.encoder = encoder_digest(model)
        self.basis = model.event_basis_digest()
        self.records = []

    def forward(self, data):
        from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire, stream_readout
        from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
        from opencood.utils import event_stream_compact as compact
        if self.training or data['object_bbx_mask'].shape[0] != 1:
            raise ValueError('Snapshot audit requires evaluation with batch size one')
        model = self.model
        _, membrane = model.encode_membrane(data['processed_lidar'], 1)
        road = model.encode_lidar(data['road_lidar'], 1).dense()
        stream = integrate_and_fire(torch.relu(model.content.project(road)), model.event_max_spikes)
        contract = MessageContract(str(data['frame_ids'][0]), str(data['road_frame_ids'][0]),
            'infrastructure', self.encoder, 'event-content32-' + self.basis,
            tuple(int(x) for x in membrane.shape[2:]), tuple(model.origin), tuple(model.step),
            tuple(int(x) for x in data['timestamps_us'][0].cpu().tolist()))
        pose = data['road_to_ego']
        meta = dict(contract=asdict(contract), source_to_ego=pose[0].cpu().tolist(),
            channels=32, max_spikes=model.event_max_spikes, schema='if-unit-events-v1')
        packets, n, used = compact.pack_prefix(stream.addresses, meta, self.budget, 256)
        assert used == sum(map(len, packets)) and used <= self.budget
        counts = stream.prefix(n)
        # Audit the original event prefix too, sharing this exact source forward.
        original = compact.CompactAccumulator(meta)
        for packet in packets:
            original.receive(packet)
        assert np.array_equal(original.finish(), counts.cpu().numpy())
        snapshot = encode_snapshot(counts.cpu().numpy(), meta)
        decoded = counts.new_tensor(decode_snapshot(snapshot, meta))
        assert torch.equal(counts, decoded)
        direct_state = stream_readout(membrane, counts, pose, contract,
            model.joint_receiver, model.content.decode)
        wire_state = stream_readout(membrane, decoded, pose, contract,
            model.joint_receiver, model.content.decode)
        assert torch.equal(direct_state, wire_state)
        direct = model.detect_bev(direct_state.flatten(1, 2))
        wire = model.detect_bev(wire_state.flatten(1, 2))
        assert all(torch.equal(direct[k], wire[k]) for k in ('psm', 'rm', 'dm'))
        name = '{:06d}.bin'.format(len(self.records))
        (self.folder / 'packets' / name).write_bytes(snapshot)
        self.records.append(dict(index=len(self.records), frame_id=contract.receiver_frame,
            reference_budget_bytes=self.budget, reference_packet_bytes=used,
            selected_unit_events=n, snapshot_bytes=len(snapshot), packet_file=name,
            packet_sha256=hashlib.sha256(snapshot).hexdigest(), counts_state_predictions_exact=True))
        return dict(wire, packet_bytes=[len(snapshot)])


def evaluate(root, budget, limit):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    protocol = read(root / 'protocol.json')
    checkpoint = Path(protocol['checkpoint'])
    config = root / ('b' + str(budget) + '.yaml')
    assert sha(checkpoint) == SOURCE_SHA == protocol['checkpoint_sha256']
    assert sha(config) == protocol['config_sha256'][str(budget)]
    torch.cuda.set_per_process_memory_fraction(.2)
    random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929)
    torch.cuda.manual_seed_all(20260929)
    cfg = load_yaml(str(config))
    assert cfg['fusion']['core_method'] == 'PairedDAIRFusion'
    assert cfg['model']['args']['event_stream']['budget_bytes'] == budget
    dataset = build_dataset(cfg, train=False)
    model = train_utils.create_model(cfg).cuda().eval()
    initial = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(initial, strict=True)
    assert model.event_packet_events == 256 and model.event_max_spikes == 8
    assert model.event_transport == 'session_zlib' and model.fusion_mode != 'ego'
    model.requires_grad_(False)
    folder = root / (('smoke_' if limit else '') + 'b' + str(budget))
    folder.mkdir(exist_ok=False); (folder / 'packets').mkdir()
    manifest = dict(checkpoint_sha256=sha(checkpoint), checkpoint=str(checkpoint),
        config_sha256=sha(config), config=str(config), seed=20260929,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
        split_sha256={k: sha(cfg[k]) for k in ('root_dir', 'validate_dir')},
        data_manifest_sha256=sha(cfg['data_manifest']), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), gpu_visible=os.getenv('CUDA_VISIBLE_DEVICES'),
        cuda_allocator_memory_fraction=.2, semantics=SEMANTICS,
        reference_budget_bytes=budget, limit=limit, training=False)
    write_json(folder / 'manifest_snapshot.json', manifest)
    wrapped = SnapshotEvaluator(model, budget, folder).eval()
    result = validate(wrapped, dataset, 2, limit=limit, include_3d=True,
        prediction_dir=folder / 'predictions', profile=False)
    assert result['samples'] == len(wrapped.records) == (limit or 1738)
    index = read(folder / 'predictions/index.json')
    for frame, record in zip(index['frames'], wrapped.records):
        assert frame['frame_id'] == record['frame_id']
        assert frame['communication']['total_bytes'] == record['snapshot_bytes']
        packet = folder / 'packets' / record['packet_file']
        assert packet.stat().st_size == record['snapshot_bytes'] and sha(packet) == record['packet_sha256']
    assert set(model.state_dict()) == set(initial)
    assert all(torch.equal(value.cpu(), initial[k]) for k, value in model.state_dict().items())
    assert sha(checkpoint) == SOURCE_SHA
    write_json(folder / 'snapshot_audit.json', dict(passed=True, frames=len(wrapped.records),
        all_state_unchanged=True, counts_state_predictions_exact=True, semantics=SEMANTICS,
        records=wrapped.records))
    result['snapshot_audit_sha256'] = sha(folder / 'snapshot_audit.json')
    result['transport_semantics'] = SEMANTICS
    write_json(folder / 'evaluation.json', result)
    print(json.dumps(dict(budget=budget, ap=result['ap'], feature_packets=result['feature_packets'])), flush=True)


def coordinate(root, gpu):
    from opencood.tools.launch_e3dsnn_joint_round import check_gpu
    status = root / 'status.json'
    try:
        # A real two-frame gate before any full split, without model updates.
        jobs = [('smoke', 4096, 2)] + [('b' + str(b), b, 0) for b in BUDGETS]
        for name, budget, limit in jobs:
            write_json(root / (name + '_admission.json'), check_gpu(gpu, True, .2))
            write_json(status, dict(state='running', stage=name, started_unix=time.time()))
            command = [sys.executable, '-u', '-m', 'opencood.tools.evaluate_e3dsnn_snapshot',
                '--root', str(root), '--evaluate', str(budget), '--limit', str(limit)]
            with (root / (name + '.log')).open('xb') as log:
                subprocess.run(command, cwd=str(REPO), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
            folder = root / (('smoke_' if limit else '') + 'b' + str(budget))
            write_json(status, dict(state='running', stage=name + '_cpu_replay', started_unix=time.time()))
            with (root / (name + '_replay.log')).open('xb') as log:
                subprocess.run([sys.executable, '-u', '-m', 'opencood.tools.replay_e3dsnn_predictions',
                    '--evaluation-dir', str(folder)], cwd=str(REPO),
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
            if limit:
                assert read(folder / 'offline_replay.json')['passed']
        results = {}
        for budget in BUDGETS:
            folder = root / ('b' + str(budget))
            row = evaluation(folder)
            audit = read(folder / 'snapshot_audit.json')
            assert audit['passed'] and audit['all_state_unchanged'] and audit['frames'] == 1738
            assert row['manifest']['checkpoint_sha256'] == SOURCE_SHA
            assert row['evaluation']['snapshot_audit_sha256'] == sha(folder / 'snapshot_audit.json')
            assert all(r['counts_state_predictions_exact'] for r in audit['records'])
            for record in audit['records']:
                assert sha(folder / 'packets' / record['packet_file']) == record['packet_sha256']
            row['reference_packet_mean_bytes'] = float(np.mean([r['reference_packet_bytes'] for r in audit['records']]))
            row['audit_sha256'] = sha(folder / 'snapshot_audit.json')
            results[str(budget)] = row
        write_json(root / 'summary.json', dict(passed=True, protocol=read(root / 'protocol.json'),
            evaluations=results, all_snapshots_state_predictions_exact=True))
        write_json(status, dict(state='complete', finished_unix=time.time(), summary_sha256=sha(root / 'summary.json')))
    except Exception:
        write_json(status, dict(state='failed', error=traceback.format_exc(), note='No automatic retry'))
        raise


def launch(root, source, gpu):
    from opencood.tools.launch_e3dsnn_joint_round import check_gpu
    checkpoint = source / 'train/best_ap70.pth'
    assert sha(checkpoint) == SOURCE_SHA
    if root.exists():
        raise FileExistsError('Fresh output root required; no duplicate launch')
    if shutil.disk_usage(str(root.parent)).free < 8 * 1024**3:
        raise RuntimeError('Insufficient disk reserve')
    admission = check_gpu(gpu, True, .2)
    root.mkdir()
    hashes = {}
    for budget in BUDGETS:
        path = root / ('b' + str(budget) + '.yaml')
        shutil.copyfile(str(source / path.name), str(path))
        hashes[str(budget)] = sha(path)
    write_json(root / 'protocol.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=SOURCE_SHA,
        config_sha256=hashes, reference_budgets=list(BUDGETS), semantics=SEMANTICS,
        seed=20260929, frames=1738, gt_boxes=28622, training=False,
        gate='two real frames plus independent CPU replay before full split',
        audit='per-frame direct/event-wire/snapshot-wire count equality, state/head equality; frozen state; CPU BEV/3D replay',
        limitations='No event-order preservation; no matched ANN control or robust cross-seed superiority claim'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root / 'pipeline.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'opencood.tools.evaluate_e3dsnn_snapshot',
            '--root', str(root), '--coordinate', '--gpu', str(gpu)], cwd=str(REPO), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch_info = dict(pid=process.pid, gpu=gpu, admission=admission, started_unix=time.time(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip())
    write_json(root / 'launch.json', launch_info)
    print(json.dumps(launch_info), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--coordinate', action='store_true')
    parser.add_argument('--evaluate', type=int, choices=BUDGETS)
    parser.add_argument('--limit', type=int, choices=(0, 2), default=0)
    args = parser.parse_args()
    if args.evaluate:
        evaluate(Path(args.root), args.evaluate, args.limit)
    elif args.coordinate:
        coordinate(Path(args.root), args.gpu)
    else:
        if not args.source: parser.error('--source required for launch')
        launch(Path(args.root), Path(args.source), args.gpu)
