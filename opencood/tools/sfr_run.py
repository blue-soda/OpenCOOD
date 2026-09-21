"""SFR component training and independent inference, with explicit frozen boundaries."""
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
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.sfr_dair import SfrDAIRDataset
from opencood.models.sfr.sender import SfrSender
from opencood.models.sfr.temporal import Temporal
from opencood.models.sfr.fusion import SfrFusion
from opencood.models.sfr.geometry import transform_boxes, transform_points, associate
from opencood.loss.point_pillar_tc_loss import PointPillarTcLoss
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def prepare(batch, sender, dataset, cfg, need_target=False):
    data = batch['ego']
    assert data['record_len'].tolist() == [2], 'DAIR v1 requires one two-agent scene per batch'
    k = data['past_k_time_interval'].numel()//2
    dense, raw = sender.extract(data['processed_lidar'], 2*k, data['anchor_box'], dataset.post_processor)
    meta = data['sfr_metadata'][0]
    messages, accounting, seen_frames = [], [], set()
    for agent, frames in enumerate(meta['observations']):
        for i, frame in enumerate(frames):
            if agent == 0 and i > 0:
                continue  # Existing reader repeats current ego in historical slots.
            key = (agent, frame['frame_id'])
            if key in seen_frames:
                continue  # One observation is transmitted once even if the sampler repeats it.
            seen_frames.add(key)
            metadata = plain(frame)
            metadata['time_s'] = metadata[cfg['time_policy']+'_time_s']
            metadata['arrival_s'] = 0.  # Independent-sample benchmark: all supplied messages arrive at query.
            metadata['time_policy'] = cfg['time_policy']
            metadata['paired_offset_ms'] = meta['paired_offset_ms']
            message, costs = sender.wire_message(raw[agent*k+i], metadata)
            transform = data['pairwise_t_matrix'][0, agent, i].to(message['boxes'])
            message['boxes'] = transform_boxes(message['boxes'], transform)
            message['points'] = transform_points(message['points'], transform)
            message['nominal_to_reference'] = transform
            messages.append(message)
            if agent:
                accounting.append(dict(frame_id=metadata['frame_id'], **costs))
    target = None
    if need_target:
        # These current infra observations are loss-only pseudo labels, never sent or updated into state.
        _, current = sender.extract(data['curr_processed_lidar'], 2, data['anchor_box'], dataset.post_processor)
        current_infra = current[1]
        transform = dense.new_tensor(meta['current_infrastructure_to_ego'])
        target = dict(boxes=transform_boxes(current_infra['boxes'], transform), scores=current_infra['scores'])
    return dense[0:1], messages, target, accounting


def motion_objective(model, messages, target, settings, prediction_mode='learned'):
    """Predict BEFORE target-time updates; frozen-detector pseudo labels explicitly identified."""
    histories = [m for m in messages if m['metadata']['agent_id'] != '0']
    terms, center_errors, counts = [], [], 0
    targets = [(m['metadata']['time_s'], m) for m in histories]
    if target is not None:
        targets.append((0., target))
    seen = set()
    for time_s, observation in sorted(targets, key=lambda item: item[0]):
        if time_s in seen:
            continue
        seen.add(time_s)
        # Loss-only replay inside an already received sample window. Arrival=0 is
        # the benchmark delivery time, not an instruction to suppress earlier
        # training spans. Only observations strictly BEFORE the target enter here.
        prefix = [dict(m, metadata=dict(m['metadata'], arrival_s=m['metadata']['time_s']))
                  for m in histories if m['metadata']['time_s'] <= time_s-settings['motion_min_dt_s']]
        if not prefix:
            continue
        tracks, _ = model.sequence(prefix, time_s, prediction_mode)
        if not tracks:
            continue
        boxes = torch.stack([t.box for t in tracks])
        keep = observation['scores'] >= settings['motion_pseudo_score']
        target_boxes = observation['boxes'][keep].detach()
        rows, cols = associate(boxes.detach(), target_boxes, 4.)
        if not len(rows):
            continue
        # Exclude ambiguous target assignments within a small center margin.
        distances = torch.cdist(boxes.detach()[:, :2], target_boxes[:, :2])
        reliable = torch.ones(len(rows), dtype=torch.bool, device=boxes.device)
        if len(target_boxes) > 1:
            sorted_distance = distances[rows].sort(dim=1).values
            reliable &= sorted_distance[:, 1]-sorted_distance[:, 0] > .75
        reliable &= torch.stack([t.score for t in tracks])[rows] >= settings['motion_pseudo_score']
        rows, cols = rows[reliable], cols[reliable]
        if not len(rows):
            continue
        center = F.smooth_l1_loss(boxes[rows, :2], target_boxes[cols, :2], reduction='none').sum(-1)
        # Frozen detector headings are only axial pseudo labels (pi ambiguity).
        yaw = 1-torch.cos(2*(boxes[rows, 6]-target_boxes[cols, 6]))
        terms.append((center+settings['motion_yaw_weight']*yaw).sum())
        center_errors.append((boxes[rows, :2]-target_boxes[cols, :2]).norm(dim=-1).detach().sum())
        counts += len(rows)
    zero = sum(p.sum()*0 for p in model.parameters())
    loss = sum(terms, zero)/max(counts, 1)
    return loss, dict(motion_valid_count=counts, motion_center_error_m=float(sum(center_errors)/counts) if counts else None,
                     supervision='frozen_single_pseudo_boxes_axial_yaw')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['motion', 'fusion'], default='fusion')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--checkpoint')
    parser.add_argument('--temporal_checkpoint')
    parser.add_argument('--temporal_mode', choices=['none', 'cv', 'learned'], default='cv')
    parser.add_argument('--max_steps', type=int, default=0)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--p', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--split_file')
    parser.add_argument('--start_index', type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.mode == 'eval' and not args.checkpoint:
        parser.error('Independent inference requires an explicit stage checkpoint')
    if args.stage == 'motion' and args.mode == 'train' and args.temporal_mode != 'learned':
        parser.error('The independent motion training stage requires --temporal_mode learned')
    if args.stage == 'fusion' and args.temporal_mode == 'learned' and not args.temporal_checkpoint:
        parser.error('Learned transport requires a separately trained temporal checkpoint')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg))
    base = load_yaml(cfg['base_config'])
    base['ectra_ego_history'] = dict(enabled=True, max_age_ms=cfg['history_max_age_ms'])
    base['train_params']['batch_size'] = 1
    if args.p is not None:
        base['binomial_p'] = args.p
    if args.split_file:
        base['root_dir' if args.mode == 'train' else 'validate_dir'] = args.split_file
    (out/'data_config.yaml').write_text(yaml.dump(base))
    random_seed = args.seed if args.seed is not None else cfg['seed' if args.mode == 'train' else 'eval_seed']
    seed(random_seed)
    dataset = SfrDAIRDataset(copy.deepcopy(base), visualize=False, train=args.mode == 'train')
    if args.start_index:
        dataset.data = dataset.data[args.start_index:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sender = SfrSender(base['model']['args'], cfg['sender']).to(device)
    sender.load_state_dict(torch.load(cfg['single_checkpoint'], map_location='cpu'), strict=True)
    sender.eval()
    temporal = Temporal(cfg['temporal']).to(device)
    if args.temporal_checkpoint:
        saved = torch.load(args.temporal_checkpoint, map_location='cpu')
        if saved['stage'] != 'motion' or saved['config']['temporal'] != cfg['temporal']:
            raise ValueError('Incompatible temporal checkpoint')
        temporal.load_state_dict(saved['model'], strict=True)
    fusion = SfrFusion(sender, cfg['fusion']).to(device)
    model = temporal if args.stage == 'motion' else fusion
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu')
        if saved['stage'] != args.stage or saved['config'] != cfg:
            raise ValueError('Checkpoint stage/config mismatch')
        model.load_state_dict(saved['model'], strict=True)
    if args.stage == 'fusion':
        temporal.requires_grad_(False).eval()
    model.train(args.mode == 'train')
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    # Use the validated baseline's detection branch, including its direction loss.
    loss_args = dict(base['loss']['args'], backbone_fix=False, aux_weights={})
    criterion = PointPillarTcLoss(loss_args).to(device)
    generator = torch.Generator().manual_seed(random_seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=args.mode == 'train', num_workers=args.workers,
                        collate_fn=dataset.collate_batch_train if args.mode == 'train' else dataset.collate_batch_test,
                        generator=generator)
    source_files = list(Path('opencood/models/sfr').glob('*.py'))+[Path(__file__), Path('opencood/data_utils/datasets/sfr_dair.py')]
    source_hashes = {str(p): digest(p) for p in source_files}
    metadata = dict(arguments=vars(args), config=cfg, started_at=time.time(), random_seed=random_seed,
                    single_sha256=digest(cfg['single_checkpoint']), checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,
                    temporal_sha256=digest(args.temporal_checkpoint) if args.temporal_checkpoint else None,
                    source_sha256=source_hashes, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                    cwd=os.getcwd(), python=sys.executable, torch=torch.__version__,
                    smoke=bool(args.max_steps or args.start_index), calibration='disabled_control_not_full_SFR',
                    split_sha256=digest(base['root_dir' if args.mode == 'train' else 'validate_dir']),
                    trainable=[n for n, p in model.named_parameters() if p.requires_grad],
                    frozen=[n for n, p in model.named_parameters() if not p.requires_grad],
                    optimizer_resume=False, supervision='pseudo' if args.stage == 'motion' else 'current_detection_GT')
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2))
    epochs = (args.epochs or cfg['training']['epochs']) if args.mode == 'train' else 1
    total_steps = 0
    for epoch in range(epochs):
        stats = {x: dict(tp=[], fp=[], gt=0, score=[]) for x in (.3, .5, .7)}
        identities, losses, valid_count, skipped, wire_bytes = [], [], 0, [], []
        motion_loss_sum, motion_error_sum = 0., 0.
        for index, batch in enumerate(loader):
            if args.max_steps and index >= args.max_steps:
                break
            if batch is None:
                skipped.append(index)
                continue
            started = time.time()
            batch = to_device(batch, device)
            ego, messages, target, accounting = prepare(batch, sender, dataset, cfg, args.stage == 'motion')
            with torch.set_grad_enabled(args.mode == 'train'):
                if args.stage == 'motion':
                    loss, diagnostics = motion_objective(temporal, messages, target, cfg['training'], args.temporal_mode)
                    valid_count += diagnostics['motion_valid_count']
                    motion_loss_sum += float(loss.detach())*diagnostics['motion_valid_count']
                    if diagnostics['motion_center_error_m'] is not None:
                        motion_error_sum += diagnostics['motion_center_error_m']*diagnostics['motion_valid_count']
                else:
                    with torch.no_grad():
                        tracks, diagnostics = temporal.sequence(messages, mode=args.temporal_mode)
                        transported = temporal.collect(tracks)
                    output, coverage = fusion(ego, transported)
                    diagnostics.update(coverage)
                    loss = criterion(output, batch['ego']['label_dict'])
                    loss = loss + sum(p.sum()*0 for p in model.parameters() if p.requires_grad)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite SFR loss')
            gradients = {}
            if args.mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                for name, module in model.named_children():
                    gradients[name] = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                if not all(np.isfinite(x) for x in gradients.values()):
                    raise FloatingPointError('Non-finite SFR gradient')
                assert all(p.grad is None for p in sender.parameters())
                assert not any(m.training for m in sender.modules())
                if args.stage == 'fusion':
                    assert all(p.grad is None for p in temporal.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                # AdamW must not update on an empty supervision set.
                if ((args.stage == 'motion' and diagnostics['motion_valid_count']) or
                    (args.stage == 'fusion' and (transported is not None or cfg['fusion']['train_heads']))):
                    optimizer.step()
            elif args.stage == 'fusion':
                corners, scores = dataset.post_processor.post_process(batch, {'ego': output})
                gt = dataset.post_processor.generate_gt_bbx(batch)
                for threshold in stats:
                    eval_utils.caluclate_tp_fp(corners, scores, gt, stats, threshold)
            meta = plain(batch['ego']['sfr_metadata'][0])
            identity = dict(frame_id=meta['vehicle_frame'], intervals=plain(batch['ego']['past_k_time_interval']),
                            timestamps=[[m['observation_us'] for m in row] for row in meta['observations']])
            identities.append(identity)
            total_bytes = sum(x['total'] for x in accounting)
            wire_bytes.append(total_bytes)
            row = dict(epoch=epoch+1, step=index, loss=float(loss.detach()), seconds=time.time()-started,
                       sample=identity, paired_offset_ms=meta['paired_offset_ms'], communication_bytes=total_bytes,
                       packet_accounting=accounting, diagnostics=diagnostics)
            if args.mode == 'train' and (args.max_steps or index < 3 or index % 100 == 0):
                row['gradients'] = gradients
            with (out/'scalars.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if index < 3 or index % 20 == 0:
                print(json.dumps(row), flush=True)
            total_steps += 1
            losses.append(row['loss'])
        if not losses or (args.stage == 'motion' and valid_count == 0):
            raise RuntimeError('No valid samples/supervision: refusing to save a nominally trained checkpoint')
        summary = dict(epoch=epoch+1, samples=len(losses), skipped_loader_indices=skipped,
                       mean_loss=sum(losses)/len(losses), motion_valid_count=valid_count,
                       motion_loss_per_instance=motion_loss_sum/valid_count if valid_count else None,
                       motion_center_error_m=motion_error_sum/valid_count if valid_count else None,
                       mean_bootstrap_communication_bytes=sum(wire_bytes)/len(wire_bytes),
                       sample_time_sha256=hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest(),
                       smoke=metadata['smoke'])
        (out/('samples_epoch%d.json' % (epoch+1))).write_text(json.dumps(identities))
        if args.mode == 'train':
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg, epoch=epoch+1,
                            stage=args.stage, interface_version='sfr-1', manifest=metadata), out/('sfr_%s_epoch%d.pth' % (args.stage, epoch+1)))
        elif args.stage == 'fusion':
            summary['ap30'], summary['ap50'], summary['ap70'] = eval_utils.eval_final_results(stats, str(out), dataset='d')
        (out/('summary_epoch%d.json' % (epoch+1))).write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    (out/'completed.json').write_text(json.dumps(dict(finished_at=time.time(), steps=total_steps)))


if __name__ == '__main__':
    main()
