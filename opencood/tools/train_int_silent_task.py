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


class ChargeDrive(TaskDrive):
    """Accumulate unsent task corrections on the ego-aligned middle lattice."""
    def __init__(self, coefficients, name, audit):
        super().__init__(coefficients, name)
        self.audit = audit
        self.charge = None

    def begin_frame(self, state, meta, shape, bounds, reason):
        if reason is not None:
            self.charge = None
        self.context = (state, meta, shape, bounds)

    def apply(self, fine, old, extra, output):
        state, meta, shape, bounds = self.context
        self.context = None
        cost = SilentDrive.apply(self, fine, old, extra, output)
        ids = extra.flatten().nonzero().flatten()
        values, shallow = samples(fine, old, ids)
        proposal = self.correction(values, shallow)
        if self.charge is None:
            aligned = torch.zeros_like(old)
        else:
            area = block_lidar_range(bounds, shape, old.shape[-2:], 4)
            aligned = warp_bev(self.charge, state['pose'], meta['pose'], area)[0]
        self.charge = aligned.masked_fill(~self.executor.skip, 0)
        flat = self.charge[0].flatten(1)
        prior = flat[:, ids].T
        if self.name == 'shuffled':
            prior = prior.roll(len(ids)//2, 0)
        total = prior + proposal
        units = (total * 4).round().clamp(-1, 1)
        sent = units * .25
        remainder = total - sent
        flat[:, ids] = remainder.T
        target = output[0].flatten(1)
        base = target[:, ids].T
        target[:, ids] = (base + sent).clamp_min(0).T
        self.emitted = int(units.count_nonzero())
        self.extra_stats = dict(charge_bytes=self.charge.numel()*self.charge.element_size(),
                                read_nonzero=int(prior.count_nonzero()),
                                changed=int((sent != (proposal*4).round()*.25).sum()))
        if self.audit:
            error = float((remainder.double()+sent.double()-prior.double()-proposal.double()).abs().max()) if len(ids) else 0.
            assert error <= 5e-8 and (not len(ids) or float(remainder.abs().max())<=.12500002)
            assert not bool(self.charge.masked_select(~self.executor.skip).count_nonzero())
            assert torch.equal(target[:,ids].T,(base+sent).clamp_min(0))
            self.extra_stats['balance_error'] = error
        # Integrating the previous charge and subtracting the emitted value.
        return cost + 21 * proposal.numel()


def projection_events(exe, events, name):
    ids = exe.refine_skip.extra.flatten().nonzero().flatten()
    flat = lambda x: x[0].flatten(1)[:, ids].T
    old = flat(exe.inputs[1])
    actual = flat(exe.result)
    prior = (old * 4).round().long()
    projected = flat(events)
    assert torch.equal(projected, (actual*4).round().long()-prior)
    base = actual if name=='drive' else exe.drive.saved[3]
    without = (base*4).round().long()-prior
    def count(value):
        return dict(values=value.numel(),addresses=int(value.count_nonzero()),ones=int((value.abs()==1).sum()),
                    units=int(value.abs().sum()))
    return dict(quiet=count(projected),whole=count(events),base_only=count(without),
                introduced=int(((without==0)&(projected!=0)).sum()),
                removed=int(((without!=0)&(projected==0)).sum()),
                changed=int((without!=projected).sum()))


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
    release = a.phase in ('release_train','release_dev')
    routing = a.phase == 'routing'
    native = routing or release or a.phase in ('native_train','native_dev')
    audit = a.phase in ('charge_train','edge_audit','native_train','release_train')
    charge_test = a.phase in ('charge_train','charge_dev')
    fit_eval = a.phase == 'fit_eval'
    if train or audit or fit_eval or routing:
        rows, traces, data_id = training_rows(a.long_base, book_path, audit)
    else:
        a.inputs = a.long_base/'inputs'
        rows, traces, data_id = complete_scenes(a, short, remainder=False)
    edge = GroupedEdge(source.bounds, 16384, book['levels'])
    names = ('continuous', 'pulse') if train else ('drive', 'continuous', 'pulse')
    native_budget = {}
    if native:
        from opencood.models.sub_modules.int_native_silent_events import NativeEventWriter
        names = ('drive','integrate','amplitude') if release else ('drive','discard','integrate','misaligned')
    if charge_test:
        names = ('pulse', 'carry', 'shuffled')
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
        if n in ('carry', 'shuffled'):
            drive = ChargeDrive(fit, n, audit)
            drive.executor = exe
        if train:
            optimizers[n] = torch.optim.Adam([drive.adapter], lr=.005)
        elif n != 'drive':
            if native:
                drive = NativeEventWriter(TaskDrive(fit, 'pulse'), exe, native_budget, n, audit)
                drive.capture_addresses = routing
                model.coarse_projector.event_observer = drive.observe_events
            drive.adapter.copy_(weights['pulse' if charge_test or native else n])
        model.coarse_executor, exe.drive = exe, drive
        if a.phase == 'edge_audit':
            def observe(events, executor=exe, name=n):
                executor.projection_record = projection_events(executor, events, name)
            model.coarse_projector.event_observer = observe
        if release and n != 'drive':
            from opencood.models.sub_modules.int_native_silent_events import SignedProjectionAccumulator
            projector = model.coarse_projector
            projector.event_linear = SignedProjectionAccumulator(projector, audit)
        models[n], drives[n] = model, drive
    data, stream = loader(rows, cfg, 0, 303)
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval() if train or fit_eval or routing else None
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
            gt = data.post.generate_gt_bbx(dict(ego=ego)) if labelled and not train and not routing else None
            row = dict(frame=meta['frame'], scene=meta['scene'], segment=meta['segment'], refresh=reason, arms={})
            end = reason is None and (depth == 1 or i+1 == len(rows) or traces[i+1]['refresh'] is not None)
            for n, model in models.items():
                old_state = states[n]
                assert model.refresh_reason(old_state, meta) == reason
                model.coarse_projector.begin(old_state, reason)
                layer.begin(old_state, reason)
                if hasattr(drives[n], 'begin_frame'):
                    drives[n].begin_frame(old_state,meta,spatial.shape[-2:],source.bounds,reason)
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
                    if native and n == 'drive':
                        mask = exe.refine_skip.extra.expand_as(exe.result)
                        delta = ((exe.result[mask]*4).round() - (exe.inputs[1][mask]*4).round())
                        native_budget['addresses'] = int(delta.count_nonzero())
                        rec['grant'] = native_budget['addresses']
                    elif native:
                        rec['native'] = drive.native
                        if release:
                            rec['projection_arithmetic'] = model.coarse_projector.event_linear.stats
                    elif n != 'drive':
                        count = rec['extra'] * 256
                        rec.update(emitted=drive.emitted, values=count, scale_ops=2*count if n!='continuous' else 0,
                                   rounding=count if n!='continuous' else 0)
                        rec['core'] += rec['scale_ops']
                        if isinstance(drive,ChargeDrive):
                            rec.update(drive.extra_stats)
                    if a.phase == 'edge_audit':
                        rec['projection'] = exe.projection_record
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
                if fit_eval and labelled and reason is None:
                    rec['task_loss'] = float(criterion(pred, ego['label_dict']))
                if gt is not None:
                    rec.update(score(data.post,ego,pred,gt,record_matches=True))
                row['arms'][n] = rec
            if reason is None:
                depth = 0 if end else depth+1
                counts = [r['executed'] for r in row['arms'].values()]
                assert all(c == counts[0] for c in counts)
            if routing and labelled and reason is None:
                from opencood.tools.eval_int_silent_middle import box_counts
                from opencood.utils import box_utils, pcd_utils
                gt_boxes = data.post.generate_gt_bbx(dict(ego=ego))
                cloud,_ = pcd_utils.read_pcd(rows[i]['pcd'])
                cloud = pcd_utils.mask_ego_points(pcd_utils.mask_points_by_range(cloud,cfg['preprocess']['cav_lidar_range']))[:,:3]
                row['points'] = [len(box_utils.get_points_in_rotated_box_3d(cloud,box)) for box in gt_boxes.cpu().numpy()]
                gate = models['drive'].coarse_executor.refine_skip
                maps = [gate.extra.long()]
                for n in names[1:]:
                    addresses = drives[n].addresses
                    assert int(addresses.sum()) == row['arms'][n]['native']['sent']
                    maps.append(addresses)
                row['box_cells'] = box_counts(gt_boxes,torch.cat(maps,1)[0].cpu().numpy(),source.bounds,spatial.shape[-2:])
            row['bytes'] = edge.stats['tile']['bytes']
            log.write(json.dumps(row,allow_nan=False)+'\n')
    assert (i+1,labels) == ((211,168) if train or fit_eval or routing else (211,0) if audit else (1014,458))
    report = dict(status='COMPLETED', scans=i+1, labels=labels, steps=steps, cross_windows=cross)
    if train:
        hook.remove()
        assert steps['pulse'] == steps['continuous'] > 0 and min(cross.values()) > 0
        torch.save({n:d.adapter.detach().cpu() for n,d in drives.items()}, a.output/'adapters.pt')
        report['weights'] = digest(a.output/'adapters.pt')
    write_json(a.output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase', choices=('fit','dev','charge_train','charge_dev','fit_eval','edge_audit','native_train','native_dev','release_train','release_dev','routing'))
    for k in ('long-base','drive','output','vehicle','fit'):
        p.add_argument('--'+k,type=Path,required=k in ('long-base','drive','output'))
    a=p.parse_args()
    if (a.phase.endswith('dev') and not a.vehicle) or (a.phase!='fit' and not a.fit):
        p.error('development needs vehicle data and frozen adapters')
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)
