"""Task-supervised silent middle corrections with paired continuous and ternary outputs."""
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_branch_fusion import BranchFusion
from opencood.models.sub_modules.int_compact_middle_update import conv_costs
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_grouped_receiver import GroupedEdge
from opencood.models.sub_modules.int_silent_drive import SilentDrive, samples
from opencood.models.sub_modules.int_silent_receiver import EdgeView
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_deep_block_online import CachedResidualPolicy, ObservedPolicy
from opencood.tools.eval_int_fusion_cache import complete_scenes
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.eval_int_online_half import build
from opencood.tools.eval_int_silent_middle import InspectMiddle, SilencePolicy, support_counts
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import compact_group_bucket, spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.run_int_silent_receiver import training_rows


class TaskDrive(SilentDrive):
    def __init__(self, coefficients, name):
        super().__init__(coefficients, 'drive')
        self.name = name
        self.adapter = torch.nn.Parameter(torch.zeros_like(self.weight))

    def correction(self, old, shallow):
        base = old * self.adapter[:, 0] + self.adapter[:, 1]
        value = torch.addmm(base, shallow, self.adapter[:, 2:].T).clamp(-.25, .25)
        if self.name == 'pulse':
            hard = value.mul(4).round().mul(.25)
            value = hard.detach() + (value - value.detach()) if torch.is_grad_enabled() else hard
        return value

    def apply(self, fine, old, extra, output):
        cost = super().apply(fine, old, extra, output)
        ids = extra.flatten().nonzero().flatten()
        state, shallow = samples(fine, old, ids)
        flat = output[0].flatten(1)
        base = flat[:, ids].T.contiguous()
        delta = self.correction(state, shallow)
        flat[:, ids] = (base + delta).clamp_min(0).T
        self.saved = (ids, state, shallow, base)
        self.emitted = int(torch.count_nonzero(delta))
        # Actual dense affine at the selected sites and a dense residual addition.
        # Comparisons, clamps, rounding and the two power-of-two scales listed separately.
        return cost + delta.numel() * 19

    def tangent(self, exe, carry, past, meta, bounds):
        ids, old, shallow, base = self.saved
        aligned = torch.zeros_like(exe.result) if carry is None else warp_bev(
            carry, past, meta['pose'], bounds)[0]
        previous = aligned[0].flatten(1)[:, ids].T
        # The frozen drive is a rectified residual with a known old-state slope.
        base_grad = previous * (1 + self.weight[:, 0]) * (base > 0)
        proposal = (base + base_grad + self.correction(old + previous, shallow)).clamp_min(0)
        selected = proposal - proposal.detach()
        tangent = aligned * exe.skip
        tangent[0].flatten(1)[:, ids] = selected.T
        return tangent


def readout_loss(core, dense_readout, middle, tangent, branches, actual, criterion, target):
    # Forward logits are the real integer/sparse detector's. The backward surrogate
    # uses the direct middle projection, with frozen fine and deep branch outputs.
    q = (middle / .25).round() * .25
    proj = core.deblocks[1](q + tangent)
    proj = branches[:, 128:256] + (proj - proj.detach())
    features = dense_readout(torch.cat((branches[:, :128], proj, branches[:, 256:]), 1))
    raw = dict(psm=core.cls_head(features), rm=core.reg_head(features))
    if core.dir_head is not None:
        raw['dm'] = core.dir_head(features)
    pred = {k: actual[k] + (v - v.detach()) for k, v in raw.items()}
    assert all(torch.equal(actual[k], v) for k, v in pred.items())
    return criterion(pred, target)


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    book_path = a.long_base / 'book_train/report.json'
    book = read(book_path)
    source, cfg, short, _, _, _, q = prepare_stream(a.long_base/'inputs', a.long_base/'clips')
    core = source.core
    source.eval().requires_grad_(False)
    train = a.phase == 'fit'
    if train:
        rows, traces, data_id = training_rows(a.long_base, book_path, False)
    else:
        a.inputs = a.long_base/'inputs'
        rows, traces, data_id = complete_scenes(a, short, remainder=False)
    edge = GroupedEdge(source.bounds, 16384, book['levels'])
    names = ('continuous', 'pulse') if train else ('drive', 'continuous', 'pulse')
    dense_readout = copy.deepcopy(core.readout) if train else None
    first = core.readout.layers[0].double_conv[0]
    layer = BranchFusion(first[0]).eval()
    first[0] = layer
    if train:
        captured = []
        hook = core.readout.register_forward_pre_hook(lambda m, value: captured.append(value[0].detach()))
    fit = read(a.drive / 'report.json')['fit']['coefficients']
    if not train:
        fit_report = read(a.fit/'report.json')
        assert fit_report['status'] == 'COMPLETED' and fit_report['weights'] == digest(a.fit/'adapters.pt')
        weights = torch.load(a.fit/'adapters.pt', map_location='cpu')
    models, drives, optimizers = {}, {}, {}
    for n in names:
        model = build(source, q, EdgeView(edge, 'tile'))
        gate = SilencePolicy(edge.arms['tile'], 'silent')
        exe = InspectMiddle(False, model.coarse_executor.receive_fine, gate)
        drive = SilentDrive(fit, 'drive') if n == 'drive' else TaskDrive(fit, n)
        if train:
            optimizers[n] = torch.optim.Adam([drive.adapter], lr=.005)
        elif n != 'drive':
            drive.adapter.copy_(weights[n])
        model.coarse_executor, exe.drive = exe, drive
        models[n], drives[n] = model, drive
    data, stream = loader(rows, cfg, 0, 303)
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval() if train else None
    costs = conv_costs(core)
    contract = dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                    data=data_id, scenes=sorted({r['scene'] for r in rows}), book=digest(book_path),
                    drive=digest(a.drive/'report.json'), params=sum(p.numel() for p in source.parameters()),
                    fitted_drive=2560, adapter=2560, seed=303, labels='ordinary only', lr=.005,
                    backward='two-frame truncated hold/silent middle and direct readout STE',
                    fit_weights=None if train else digest(a.fit/'adapters.pt'))
    write_json(a.output/'contract.json', contract)
    steps = dict.fromkeys(names, 0)
    cross = dict.fromkeys(names, 0)
    labels = 0
    factory = partial(CachedResidualPolicy, compiler=compact_group_bucket)
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for i, (sample, trace) in enumerate(zip(stream, traces)):
            meta, reason = sample['meta'], trace['refresh']
            assert meta['frame'] == trace['frame']
            if i == 0 or meta['segment'] != rows[i-1]['segment']:
                states = dict.fromkeys(names, None if reason=='initial' else dict(scene=''))
                selectors = {n: ObservedPolicy(models[n], factory) for n in names}
            if reason is not None:
                depth = 0
                carry = dict.fromkeys(names)
                previous_sites = dict.fromkeys(names)
                losses = {n: [] for n in names}
            ego = to_device(sample['ego'], torch.device('cuda'))
            spatial = spatial_input(source, ego)
            edge.start(meta, spatial.shape[-2:], reason)
            labelled = meta['supervised']
            labels += int(labelled)
            gt = data.post.generate_gt_bbx(dict(ego=ego)) if labelled and not train else None
            row = dict(frame=meta['frame'], scene=meta['scene'], segment=meta['segment'], refresh=reason, arms={})
            end = reason is None and (depth == 1 or i+1 == len(rows) or traces[i+1]['refresh'] is not None)
            for n, model in models.items():
                old_state = states[n]
                assert model.refresh_reason(old_state, meta) == reason
                model.coarse_projector.begin(old_state, reason)
                layer.begin(old_state, reason)
                selector = selectors[n]
                selector.observation = None
                (pred, state, info), base_cost = measured_step(model, spatial, old_state, meta, selector)
                assert reason == info['full_refresh']
                state['branch_fusion'] = layer.new
                states[n] = state
                rec = dict(core=base_cost + layer.ledger['float_conv'])
                if train:
                    assert len(captured) == 1
                    branches = captured.pop()
                if reason is None:
                    exe, drive = model.coarse_executor, drives[n]
                    assert exe.counts == support_counts(~exe.skip)
                    rec.update(executed=exe.counts, extra=int(exe.refine_skip.extra.sum()), drive_ops=exe.drive_ops)
                    rec['core'] += exe.drive_ops + sum(costs[k]*v for k,v in exe.counts.items())
                    if n != 'drive':
                        count = rec['extra'] * 256
                        rec.update(emitted=drive.emitted, values=count, scale_ops=2*count if n=='pulse' else 0,
                                   rounding=count if n=='pulse' else 0)
                        rec['core'] += rec['scale_ops']
                    if train:
                        with torch.enable_grad():
                            area = block_lidar_range(source.bounds,spatial.shape[-2:],exe.result.shape[-2:],4)
                            previous = carry[n]
                            tangent = drive.tangent(exe, previous, old_state['pose'], meta, area)
                            if labelled:
                                loss = readout_loss(core,dense_readout,state['coarse'],tangent,branches,
                                                    pred,criterion,ego['label_dict'])
                                losses[n].append(loss)
                                rec['loss'] = float(loss.detach())
                                if previous is not None:
                                    g, = torch.autograd.grad(loss, previous,retain_graph=True)
                                    rec['past_grad_l1'] = float(g.flatten(2)[:, :, previous_sites[n]].abs().sum())
                                    cross[n] += int(rec['past_grad_l1'] > 0)
                            carry[n] = tangent
                            previous_sites[n] = drive.saved[0]
                            if end and losses[n]:
                                opt = optimizers[n]
                                opt.zero_grad(set_to_none=True)
                                torch.stack(losses[n]).mean().backward()
                                norm = torch.nn.utils.clip_grad_norm_([drive.adapter],5.)
                                assert torch.isfinite(norm)
                                opt.step()
                                rec['grad_norm'] = float(norm)
                                steps[n] += 1
                        if end:
                            carry[n] = None
                            losses[n].clear()
                    exe.inputs = exe.result = None
                if gt is not None:
                    rec.update(score(data.post,ego,pred,gt,record_matches=True))
                row['arms'][n] = rec
            if reason is None:
                depth = 0 if end else depth+1
                counts = [r['executed'] for r in row['arms'].values()]
                assert all(c == counts[0] for c in counts)
            row['bytes'] = edge.stats['tile']['bytes']
            log.write(json.dumps(row,allow_nan=False)+'\n')
    assert (i+1,labels) == ((211,168) if train else (1014,458))
    report = dict(status='COMPLETED', scans=i+1, labels=labels, steps=steps, cross_windows=cross)
    if train:
        hook.remove()
        assert steps['pulse'] == steps['continuous'] > 0 and min(cross.values()) > 0
        torch.save({n:d.adapter.detach().cpu() for n,d in drives.items()}, a.output/'adapters.pt')
        report['weights'] = digest(a.output/'adapters.pt')
    write_json(a.output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase', choices=('fit','dev'))
    for k in ('long-base','drive','output','vehicle','fit'):
        p.add_argument('--'+k,type=Path,required=k in ('long-base','drive','output'))
    a=p.parse_args()
    if a.phase=='dev' and (not a.vehicle or not a.fit):
        p.error('development needs vehicle data and frozen adapters')
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)
