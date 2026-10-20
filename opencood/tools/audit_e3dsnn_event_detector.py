"""Bounded real-data E0 audit; discarded optimizer step, no AP/training claim."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
import yaml
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import checked_forward
from opencood.tools.launch_e3dsnn_hybrid_round import sha
from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire, stream_readout
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.utils import event_stream_packet as wire


def audit(root, base):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    source = Path(base) / 'e3dsnn_joint_round1_20261019'
    checkpoint = source / 'joint_adapt10/best_ap70.pth'
    if sha(checkpoint) != 'f4a5795b8b001a7cc4c04fd9eb3959f02df70181afe23691e733cb3aa4eb8588':
        raise ValueError('Unexpected registered source checkpoint')
    raw = yaml.safe_load((source / 'joint/p3.yaml').read_text())
    raw['model']['core_method'] = 'e3dsnn_event_detector'
    raw['model']['args'].update(joint_prefix_weights=None, joint_rate_weight=0.,
        content_full_packet=False, packet_roundtrip=False,
        event_stream=dict(budget_bytes=4096, train_budget_range=[4096,8192],
                          packet_events=256, max_spikes=8))
    (root / 'config.yaml').write_text(yaml.safe_dump(raw))
    cfg = load_yaml(str(root / 'config.yaml'))
    random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929)
    dataset = build_dataset(cfg, train=True)
    samples = [dataset[i] for i in (0,1)]
    batch = train_utils.to_device(dataset.collate_batch_train(samples), torch.device('cuda'))
    model = train_utils.create_model(cfg).cuda().eval()
    reference = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(reference, strict=True)
    data = batch['ego']
    report = dict(started_unix=time.time(), source=str(checkpoint),
        source_sha256=sha(checkpoint), code_commit=subprocess.check_output(
            ['git','rev-parse','HEAD']).decode().strip(),
        frames=data['frame_ids'], tests=[], training_started=False,
        status='audit_only_no_AP', trainable=[k for k,p in model.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert report['trainable'] == ['content.project.weight']
    original_encode, original_membrane = model.encode_lidar, model.encode_membrane
    with torch.no_grad():
        ego, membrane = original_membrane(data['processed_lidar'], 2)
        road = original_encode(data['road_lidar'], 2)
        model.encode_membrane = lambda lidar,n: (ego,membrane)
        model.encode_lidar = lambda lidar,n: road
        for budget in (0,4096,8192):
            model.event_budget = budget; model.packet_roundtrip = False
            direct = model(data)
            model.packet_roundtrip = True; decoded = model(data)
            assert all(torch.equal(direct[k],decoded[k]) for k in ('psm','rm','dm'))
            assert all(n<=budget for n in decoded['packet_bytes'])
            repeated = model(data)
            assert all(torch.equal(repeated[k],decoded[k]) for k in ('psm','rm','dm'))
            report['tests'].append(dict(budget=budget, bytes=decoded['packet_bytes'],
                events=decoded['stream_events'], direct_wire_exact=True, repeated_exact=True))
        model.event_budget = 0; silent = model(data)
        model.fusion_mode = 'ego'; local = model(data); model.fusion_mode = 'residual'
        assert all(torch.equal(silent[k],local[k]) for k in ('psm','rm','dm'))
        report['zero_budget_is_ego_exact'] = True
        old_cfg = load_yaml(str(source/'joint/p3.yaml'))
        old = train_utils.create_model(old_cfg).cuda().eval()
        old.load_state_dict(reference, strict=True); old.fusion_mode='ego'
        old.encode_membrane=lambda lidar,n:(ego,membrane)
        old_local=old(data)
        assert all(torch.equal(local[k],old_local[k]) for k in ('psm','rm','dm'))
        report['ego_matches_source_exact'] = True
        del old
        report['same_events_repacking'] = []
        for i in range(2):
            drive = torch.relu(model.content.project(road.dense()[i:i+1]))
            stream = integrate_and_fire(drive, model.event_max_spikes)
            c = MessageContract(str(data['frame_ids'][i]), str(data['road_frame_ids'][i]),
                'infrastructure', model._encoder_identity, 'event-content32-'+model.event_basis_digest(),
                tuple(int(n) for n in membrane.shape[2:]), tuple(model.origin), tuple(model.step),
                tuple(int(n) for n in data['timestamps_us'][i].cpu().tolist()))
            pose = data['road_to_ego'][i:i+1]
            meta = dict(contract=asdict(c), source_to_ego=pose[0].cpu().tolist(), channels=32,
                        max_spikes=model.event_max_spikes, schema='if-unit-events-v1')
            n = min(512, len(stream.addresses))
            expected = None; sizes = []
            for capacity in (64,256,1024):
                packets, received, used = wire.pack_prefix(stream.addresses[:n],meta,2**30,capacity)
                acc = wire.StreamAccumulator(meta)
                for packet in packets:
                    acc.receive(packet)
                counts = drive.new_tensor(acc.finish())
                assert torch.equal(counts,stream.prefix(n)) and received==n
                state = stream_readout(membrane[i:i+1],counts,pose,c,
                                       model.joint_receiver,model.content.decode)
                detected = model.detect_bev(state.flatten(1,2))
                if expected is not None:
                    assert all(torch.equal(expected[k],detected[k]) for k in ('psm','rm','dm'))
                expected = detected
                sizes.append(dict(packet_events=capacity,packets=len(packets),bytes=used))
            report['same_events_repacking'].append(dict(frame=data['frame_ids'][i],
                events=n,variants=sizes,counts_and_predictions_exact=True,
                candidate_events=len(stream.addresses),
                saturated_latent_entries=int((drive>=model.event_max_spikes).sum())))
    model.encode_lidar, model.encode_membrane = original_encode, original_membrane
    model.packet_roundtrip = False; model.event_budget = 4096
    model.train(); model.message_dropout = 0.
    criterion = train_utils.create_loss(cfg).cuda()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    optimizer.zero_grad(set_to_none=True)
    output, loss = checked_forward(model,criterion,batch)
    loss.backward()
    gradients = {k:float(p.grad.abs().sum()) for k,p in model.named_parameters() if p.grad is not None}
    assert set(gradients)==set(report['trainable'])
    assert all(np.isfinite(v) and v>0 for v in gradients.values())
    optimizer.step()
    actual = model.state_dict()
    frozen = [k for k in actual if k!='content.project.weight']
    assert all(torch.equal(actual[k].cpu(),reference[k]) for k in frozen)
    assert not torch.equal(actual['content.project.weight'].cpu(),reference['content.project.weight'])
    report.update(passed=True, gradient_l1=gradients, loss=float(loss.detach()),
        sampled_budgets=output['stream_budgets'], sampled_bytes=output['packet_bytes'],
        all_other_parameters_and_buffers_unchanged=True, optimizer_step_discarded=True)
    (root/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--base',default='/data1/tzm/workspace/diagnostics')
    args=parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(.2)
    audit(args.root,args.base)
