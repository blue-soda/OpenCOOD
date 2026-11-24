"""Train a three-scale signed-event residual backbone on chronological DAIR clips."""
import argparse
import inspect
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from opencood.data_utils.int_ego_stream import training_order
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_translation_anchor import TranslationAnchor
from opencood.tools.check_int_deep_spike_stream import build_source, read
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.utils.eval_utils import calculate_ap
from opencood.utils import box_utils, pcd_utils


class Sender(nn.Module):
    def __init__(self, channels, cap):
        super().__init__()
        self.log_step = nn.Parameter(torch.full((1, channels, 1, 1), math.log(.25)))
        self.cap = cap
        self.step_multiplier = 1.
        self.continuous_threshold = None
        self.payload_dtype = None
        self.int8_ste = False
        self.full_amplitude = False
        self.drop_silent_remainder = False
        self.sending_reference = None

    def step(self):
        return self.log_step.exp().clamp(.02, 4.) * self.step_multiplier

    def forward(self, target, prior):
        step = self.step()
        if self.full_amplitude:
            # An independently reset spatial ANN passes the original activation to the next convolution.
            change = target - prior
            stats = torch.stack((change.ne(0).float().mean(), change.new_zeros(()),
                                 (change.abs() / step).mean())).detach()
            return change, target, stats
        reference = prior
        if self.drop_silent_remainder and torch.is_tensor(prior):
            reference = self.sending_reference
        residual = (target - reference) / step
        clipped = residual.clamp(-self.cap, self.cap)
        if self.continuous_threshold is not None:
            active = residual.abs() > self.continuous_threshold
            amplitude = clipped * step
            if self.payload_dtype == 'int8':
                levels = clipped * (127. / self.cap)
                code = levels.round().to(torch.int8).to(target.dtype)
                if self.int8_ste and torch.is_grad_enabled():
                    code = code + (levels - levels.detach())
                amplitude = code * (step * (self.cap / 127.))
            elif self.payload_dtype == 'fp16':
                amplitude = amplitude.half().to(target.dtype)
            change = torch.where(active, amplitude, torch.zeros_like(target))
            stats = torch.stack((active.float().mean(), (residual.abs() > self.cap).float().mean(),
                                 (change / step).abs().mean())).detach()
        else:
            code = clipped.round()
            change = (code.detach() + (clipped - clipped.detach())) * step
            stats = torch.stack(((code != 0).float().mean(), (residual.abs() > self.cap + .5).float().mean(),
                                 code.abs().mean())).detach()
        if self.drop_silent_remainder:
            # Only the sender discards silent differences. The receiver still sees the emitted events.
            self.sending_reference = torch.where(change == 0, target, reference + change).detach()
        return change, prior + change, stats


def delta_conv(conv, events, first):
    return F.conv2d(events, conv.weight, conv.bias if first else None, conv.stride,
                    conv.padding, conv.dilation, conv.groups)


class SpikeBlock(nn.Module):
    def __init__(self, block, cap):
        super().__init__()
        self.conv1 = fuse_conv_bn_eval(block.conv1, block.bn1)
        self.conv2 = fuse_conv_bn_eval(block.conv2, block.bn2)
        self.skip = fuse_conv_bn_eval(*block.downsample) if block.downsample is not None else None
        self.sender1 = Sender(block.conv1.out_channels, cap)
        self.sender2 = Sender(block.conv2.out_channels, cap)

    def forward(self, events, state):
        first = state is None
        pre1, sent1, pre2, sent2 = (0., 0., 0., 0.) if first else state
        pre1 = pre1 + delta_conv(self.conv1, events, first)
        internal, sent1, rate1 = self.sender1(pre1.relu(), sent1)
        shortcut = events if self.skip is None else delta_conv(self.skip, events, first)
        pre2 = pre2 + delta_conv(self.conv2, internal, first) + shortcut
        outgoing, sent2, rate2 = self.sender2(pre2.relu(), sent2)
        return outgoing, (pre1, sent1, pre2, sent2), (rate1 + rate2) / 2


class SpikePyramid(nn.Module):
    def __init__(self, core, cap, stage_caps=None):
        super().__init__()
        self.input_sender = Sender(64, cap)
        budgets = stage_caps if stage_caps is not None else [cap] * 3
        self.stages = nn.ModuleList(nn.ModuleList(SpikeBlock(block, limit) for block in stage)
                                    for limit, stage in zip(budgets, (core.observation,
                                                                      core.coarse_encoder, core.deep_encoder)))
        self.requires_grad_(True)

    def forward(self, spatial, core, state, crop=None):
        old_input, old_blocks = (0., {}) if state is None else state
        events, received, _ = self.input_sender(spatial, old_input)
        states, outputs, rates = {}, [], []
        for i, stage in enumerate(self.stages):
            stage_rates = []
            for j, block in enumerate(stage):
                key = (i, j)
                events, states[key], rate = block(events, old_blocks.get(key))
                stage_rates.append(rate)
            branch = core.deblocks[i](states[key][3])
            outputs.append(branch if crop is None else branch[..., crop[0], crop[1]])
            rates.append(torch.stack(stage_rates).mean(0))
        readout = core.readout(torch.cat(outputs, dim=1))
        pred = dict(psm=core.cls_head(readout), rm=core.reg_head(readout))
        if core.dir_head is not None:
            pred['dm'] = core.dir_head(readout)
        return pred, (received, states), torch.stack(rates)


def detach_state(state):
    received, blocks = state
    return received.detach(), {k: tuple(v.detach() for v in values) for k, values in blocks.items()}


@torch.no_grad()
def continuous_reference(rows, cfg, source, include_unlabeled=False):
    """Same spatial stages and frozen readout with continuous activations."""
    ds, stream = loader(rows, cfg, 0, 303)
    core = source.core
    stats = {t:dict(tp=[], fp=[], gt=0, score=[]) for t in (.5, .7)}
    for sample in stream:
        supervised = sample['meta']['supervised']
        if not supervised and not include_unlabeled:
            continue
        ego = to_device(sample['ego'], torch.device('cuda'))
        value = spatial_input(source, ego)
        branches = []
        for stage, project in zip((core.observation, core.coarse_encoder, core.deep_encoder), core.deblocks):
            value = stage(value)
            branches.append(project(value))
        features = core.readout(torch.cat(branches, dim=1))
        pred = dict(psm=core.cls_head(features), rm=core.reg_head(features))
        if core.dir_head is not None:
            pred['dm'] = core.dir_head(features)
        if not supervised:
            continue
        measured = score(ds.post, ego, pred, ds.post.generate_gt_bbx(dict(ego=ego)))['ap']
        for threshold, counts in measured.items():
            for key in stats[threshold]:
                stats[threshold][key] += counts[key]
    return {str(t):calculate_ap(stats,t)[0] for t in stats}


def record(path, data):
    with path.open('a') as stream:
        stream.write(json.dumps(data, allow_nan=False) + '\n')


def checkpoint(path, net, opt, epoch, best, stale):
    temp = path.with_suffix('.tmp')
    torch.save(dict(net=net.state_dict(), optimizer=opt.state_dict(), epoch=epoch,
                    best=best, stale=stale), temp)
    os.replace(str(temp), str(path))


def fine_target_loss(core, spatial, received, labels):
    """During training, match the continuous fine feature at positive anchor cells."""
    mask = labels['pos_equal_one'].any(dim=-1)
    assert mask.shape == (received.shape[0], *received.shape[-2:])
    if not mask.any():
        return received.new_zeros(())
    with torch.no_grad():
        target = core.observation(spatial).permute(0,2,3,1)[mask]
        scale = target.abs().mean().clamp_min(.25)
    return F.smooth_l1_loss(received.permute(0,2,3,1)[mask], target) / scale


def replay(rows, cfg, source, net, criterion, optimizer, epoch, out, point_bins, memoryless=False,
           fine_distill_weight=0., anchor_mode=None, order_seed=303, frame_log=None, window_mode=None,
           target_log=None):
    train = optimizer is not None
    if train:
        rows = training_order(rows, length=5, seed=order_seed + epoch)
    if window_mode:
        from opencood.models.sub_modules.int_planar_window import attach_window
        rows = attach_window(rows,window_mode=='rigid')
    ds, stream = loader(rows, cfg, 0, 303)
    stats = {t:dict(tp=[], fp=[], gt=0, score=[]) for t in (.5, .7)}
    state, pending, total, labels, updates = None, [], 0., 0, 0
    rates = torch.zeros(3, 3, device='cuda')
    weak_hits = weak_total = 0
    imitation_total = 0.
    anchor = TranslationAnchor(cfg['preprocess']['args']['voxel_size'][:2]) if anchor_mode else None
    resets = placements = 0
    outside = observed = moved = 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    for i, sample in enumerate(stream):
        row = rows[i]
        assert sample['meta']['frame'] == row['frame']
        reset = (i == 0 or row.get('reset_before') or row['segment'] != rows[i-1]['segment'])
        if reset:
            state, age = None, 0
        age += 1
        end = i + 1 == len(rows) or rows[i+1].get('reset_before') or rows[i+1]['segment'] != row['segment']
        ego = to_device(sample['ego'], torch.device('cuda'))
        with torch.no_grad():
            spatial = spatial_input(source, ego)
            network_input, crop = spatial, None
            if anchor is not None:
                restart, shift = anchor.locate(row['pose'], reset)
                if restart:
                    state = None
                    resets += 1
                if anchor_mode != 'p5':
                    placement = shift if anchor_mode == 'translation' else (0, 0)
                    placements += int(any(placement))
                    network_input, crop = anchor.place(spatial, placement)
        with torch.set_grad_enabled(train and (not memoryless or row['supervised'])):
            # Keep the clip and optimizer boundaries identical for the independent-frame arm.
            initial_state = None if memoryless else state
            if window_mode:
                assert crop is None
                transform = row['window_transform']
                moved += int(transform is not None)
                outside += sample['meta'].get('outside_window',0)
                observed += sample['meta'].get('before_alignment',sample['meta']['point_count_after_crop'])
                pred, state, activity = net(network_input,source.core,initial_state,
                                            ordinary=row['window_ordinary'],transform=transform)
            else:
                pred, state, activity = (net(network_input, source.core, initial_state) if crop is None
                                         else net(network_input, source.core, initial_state, crop))
            rates += activity
            if row['supervised']:
                loss = criterion(pred, ego['label_dict'])
                if not torch.isfinite(loss):
                    raise FloatingPointError((row['frame'], 'loss'))
                total += loss.item()
                labels += 1
                if train:
                    if fine_distill_weight:
                        last_fine_block = (0,len(net.stages[0])-1)
                        received = state[1][last_fine_block][3]
                        if crop is not None:
                            received = received[..., crop[0], crop[1]]
                        teacher_loss = fine_target_loss(source.core, spatial, received, ego['label_dict'])
                        if not torch.isfinite(teacher_loss):
                            raise FloatingPointError((row['frame'], 'fine target'))
                        imitation_total += teacher_loss.item()
                        loss = loss + fine_distill_weight * teacher_loss
                    if memoryless:
                        # Independent graphs can be released immediately. Normalize the sum and
                        # update parameters only at the same clip boundary as the temporal arm.
                        loss.backward()
                    else:
                        pending.append(loss)
                else:
                    gt = ds.post.generate_gt_bbx(dict(ego=ego))
                    scored = score(ds.post, ego, pred, gt, record_matches=True)
                    measured = scored['ap']
                    if row['frame'] not in point_bins:
                        cloud = ds.read_points(row)
                        cloud = pcd_utils.mask_ego_points(pcd_utils.mask_points_by_range(
                            cloud, cfg['preprocess']['cav_lidar_range']))[:, :3]
                        point_bins[row['frame']] = [6 <= len(box_utils.get_points_in_rotated_box_3d(cloud, b)) <= 20
                                                    for b in gt.cpu().numpy()]
                    weak = point_bins[row['frame']]
                    weak_total += sum(weak)
                    frame_hits = sum(weak[k] for k in scored['hit_ids'])
                    weak_hits += frame_hits
                    if target_log is not None:
                        assert row.get('dataset') == 'v2v_single'
                        # Match the scored, range-filtered GT back to its single-agent annotation.
                        # The postprocessor may reorder objects when removing duplicates.
                        values = ego['object_bbx_center'][ego['object_bbx_mask'] == 1]
                        corners = box_utils.boxes_to_corners_3d(values, ds.post.params['order'])
                        corners = box_utils.project_box3d(corners.float(), ego['transformation_matrix_clean'])
                        same = (gt[:,None] - corners[None,:]).abs().amax(dim=(-1,-2)) < 1e-4
                        assert bool((same.sum(1) == 1).all())
                        indices = same.nonzero(as_tuple=False)[:,1].tolist()
                        ids = [str(ego['object_ids'][j]) for j in indices]
                        assert len(ids) == len(set(ids))
                        matched = set(scored['hit_ids'])
                        record(target_log, dict(scene=row['scene'], source_frame=row['source_frame'],
                            frame=row['frame'], reset=bool(row['reset_before']), ids=ids,
                            centers=gt.mean(1)[:,:2].tolist(), hits=[j in matched for j in range(len(gt))],
                            weak=weak))
                    if frame_log is not None:
                        record(frame_log, dict(scene=row['scene'], frame=row['frame'],
                               weak=sum(weak), hits=frame_hits, ap70=measured[.7]))
                    for threshold, counts in measured.items():
                        for key in stats[threshold]:
                            stats[threshold][key] += counts[key]
        if memoryless:
            state = None
        if train and not memoryless and (age % 2 == 0 or end):
            if pending:
                torch.stack(pending).sum().backward()
                pending.clear()
            state = detach_state(state)
        if train and end:
            count = sum(r['supervised'] for r in rows[i-age+1:i+1])
            for p in net.parameters():
                if p.grad is not None:
                    p.grad.div_(count)
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            stage_grad = [float(torch.stack([p.grad.norm() for p in stage.parameters()
                                             if p.grad is not None]).norm()) for stage in net.stages]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            record(out/'updates.jsonl', dict(epoch=epoch, update=updates, grad=float(grad),
                   stage_grad_after_clip=stage_grad, loss=total/labels, rates=(rates/(i+1)).tolist(),
                   frame=row['frame']))
    result = dict(scans=len(rows), labels=labels, updates=updates, loss=total/labels,
                  rates=(rates/len(rows)).tolist())
    if anchor is not None:
        result.update(anchor_resets=resets, shifted_scans=placements)
    if window_mode:
        result.update(window_mode=window_mode,transformed_scans=moved,observed_points=observed,
                      points_outside_window=outside)
    if train and fine_distill_weight:
        result['positive_fine_imitation_loss'] = imitation_total / labels
    if not train:
        result.update(ap50=calculate_ap(stats,.5)[0], ap70=calculate_ap(stats,.7)[0],
                      weak_hits=weak_hits, weak_total=weak_total)
    return result


def run(a):
    torch.manual_seed(303)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = load_yaml(str(a.inputs/'config.yaml'))
    cfg['data_augment'] = []
    plan = read(a.plan)
    receipt = read(a.plan.parent/'report.json')
    assert digest(a.plan) == receipt['plan']
    fit = receipt['splits']['fit']['scenes']
    monitor = fit[-2:]
    training = fit[:4] if a.phase == 'short' else fit[:-2]
    rows = plan['rows']['fit']
    train = [r for r in rows if r['scene'] in training]
    dev = [r for r in rows if r['scene'] in monitor]
    assert train and dev and not set(training).intersection(monitor)
    for r in train + dev:
        assert r['split'] == 'train'
        for key in ('pcd','label'):
            if r[key]:
                assert digest(Path(r[key])) == plan['members'][r[key]]
    source = build_source(a.inputs, cfg)
    net = SpikePyramid(source.core, a.cap, a.stage_caps).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=.0001)
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    if a.start:
        net.load_state_dict(torch.load(a.start, map_location='cpu')['net'])
    contract = dict(plan=digest(a.plan), script=digest(Path(__file__)), phase=a.phase,
                    training=training, monitor=monitor, cap=a.cap, epochs=a.epochs, lr=a.lr,
                    initial=digest(a.start) if a.start else None, trainable=sum(p.numel() for p in net.parameters()),
                    rate_columns=['nonzero_fraction','overflow_fraction','mean_absolute_count'],
                    fit_clip=5, bptt=2, validation='whole segments; no periodic refresh; fixed ego grid')
    if a.stage_caps is not None:
        contract['stage_caps'] = a.stage_caps
    if a.memoryless:
        contract.update(memoryless=True, validation='independent frames; shared clip and optimizer schedule')
    if a.anchor_mode:
        contract.update(anchor_mode=a.anchor_mode, anchor_code=digest(Path(inspect.getfile(TranslationAnchor))),
            validation='causal five-scan anchors, reset on segment or margin overflow')
    if a.fine_distill_weight:
        if not math.isfinite(a.fine_distill_weight) or a.fine_distill_weight < 0:
            raise ValueError('fine imitation weight must be positive and finite')
        contract['fine_distill_weight'] = a.fine_distill_weight
    best, stale, first = -1., 0, 0
    if a.resume:
        assert read(a.output/'contract.json') == contract
        saved = torch.load(a.output/'latest.pt', map_location='cuda')
        net.load_state_dict(saved['net'])
        opt.load_state_dict(saved['optimizer'])
        best, stale, first = saved['best'], saved['stale'], saved['epoch'] + 1
        if (a.output/'report.json').exists() or first > a.epochs or stale >= 4:
            raise ValueError('This checkpoint has reached its stopping condition')
    else:
        write_json(a.output/'contract.json', contract)
        write_json(a.output/'continuous_reference.json', continuous_reference(dev, cfg, source))
    point_bins = {}
    for epoch in range(first, a.epochs+1):
        fitted = replay(train,cfg,source,net,criterion,opt,epoch,a.output,point_bins,
                        a.memoryless,a.fine_distill_weight,a.anchor_mode) if epoch else None
        valid = replay(dev,cfg,source,net,criterion,None,epoch,a.output,point_bins,a.memoryless,
                       anchor_mode=a.anchor_mode)
        if any(not 0 < layer[0] < 1 for layer in valid['rates']):
            raise RuntimeError(('inactive or all-active stage', valid['rates']))
        improved = valid['ap70'] > best
        if improved:
            best, stale = valid['ap70'], 0
            checkpoint(a.output/'best.pt',net,opt,epoch,best,stale)
        else:
            stale += 1
        checkpoint(a.output/'latest.pt',net,opt,epoch,best,stale)
        event = dict(epoch=epoch,train=fitted,validation=valid,best_ap70=best,stale=stale)
        record(a.output/'epochs.jsonl',event)
        print(json.dumps(event), flush=True)
        if a.phase == 'long' and stale >= 4:
            break
    write_json(a.output/'report.json', dict(epoch=epoch,best_ap70=best,stale=stale,
                                          stop='patience' if stale >= 4 else 'budget'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['short','long'], required=True)
    p.add_argument('--start', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--memoryless', action='store_true')
    p.add_argument('--anchor-mode', choices=['p5','fixed','translation'])
    p.add_argument('--fine-distill-weight', type=float, default=0.)
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--cap', type=int, choices=[1,2,4], default=4)
    p.add_argument('--stage-caps', nargs=3, type=int, choices=[1,2,4])
    p.add_argument('--lr', type=float, default=.0001)
    run(p.parse_args())
