"""Matched long training of native signed-count and periodic exact-refresh BEV pyramids."""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.audit_int_spike_pyramid_ops import ConvLedger
from opencood.models.sub_modules.int_planar_window import current_readout


class RecoveryPyramid(fit.SpikePyramid):
    def __init__(self, core, initial, variant, bounds=None):
        super().__init__(core, 32, [32, 16, 16])
        self.load_state_dict(torch.load(initial, map_location='cpu')['net'])
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, fit.Sender):
                    module.log_step.sub_(math.log(4))
        self.heads = nn.ModuleDict({k: copy.deepcopy(getattr(core, k)) for k in
                                    ('cls_head', 'reg_head', 'dir_head') if getattr(core, k) is not None})
        self.heads.requires_grad_(True)
        self.variant = variant
        self.refreshes = 0
        self.bounds = bounds
        self.ordinary_multiplier = 2.
        self.warp_values = 0
        self.senders = tuple(m for m in self.modules() if isinstance(m,fit.Sender))
        self.fine_senders = (self.input_sender, *(m for m in self.stages[0].modules() if isinstance(m,fit.Sender)))
        self.keep_fine_precision = variant in ('fine_temporal','fine_independent')

    def forward(self, spatial, core, state, ordinary=False, transform=None):
        readout = core.readout
        support = None
        if self.variant in ('coarse_fixed','coarse_rigid','fine_temporal','fine_independent'):
            for sender in self.senders:
                sender.step_multiplier = self.ordinary_multiplier if ordinary else 1.
            if self.keep_fine_precision and ordinary:
                for sender in self.fine_senders:
                    sender.step_multiplier = 1.
            if transform is not None:
                def readout(x):
                    nonlocal support
                    x = core.readout(x)
                    packed = torch.cat((x,torch.ones_like(x[:,:1])),dim=1)
                    self.warp_values += packed.numel()
                    sampled = current_readout(packed,transform,self.bounds)
                    support = sampled[:,-1:]
                    return sampled[:,:-1]
        decoder = SimpleNamespace(deblocks=core.deblocks, readout=core.readout,
                                  cls_head=self.heads['cls_head'], reg_head=self.heads['reg_head'],
                                  dir_head=self.heads['dir_head'] if 'dir_head' in self.heads else None)
        if transform is not None:
            decoder.readout = readout
        if state is not None or self.variant != 'refresh':
            pred, memory, rates = super().forward(spatial, decoder, state)
            if support is not None:
                # Zero-padded latent features otherwise score at the learned head bias
                # (~0.5 probability here). Interpolate a background logit off the map.
                assert decoder.cls_head.kernel_size == (1,1)
                bias = decoder.cls_head.bias.view(1,-1,1,1)
                pred['psm'] = pred['psm'] + (1-support).clamp(0,1)*(-8.-bias)
            return pred, memory, rates
        # One exact spatial pass initializes *both* convolution accumulators and sent levels.
        # Subsequent signed changes thus update the very same learned convolution weights.
        self.refreshes += 1
        states, branches = {}, []
        x = spatial
        for i, stage in enumerate(self.stages):
            for j, block in enumerate(stage):
                pre1 = block.conv1(x)
                sent1 = pre1.relu()
                shortcut = x if block.skip is None else block.skip(x)
                pre2 = block.conv2(sent1) + shortcut
                x = pre2.relu()
                states[i,j] = pre1, sent1, pre2, x
            branches.append(core.deblocks[i](x))
        features = core.readout(torch.cat(branches, dim=1))
        pred = dict(psm=decoder.cls_head(features), rm=decoder.reg_head(features))
        if decoder.dir_head is not None:
            pred['dm'] = decoder.dir_head(features)
        return pred, (spatial, states), spatial.new_zeros(3,3)


def periodic(rows):
    result = []
    age = 0
    for i, row in enumerate(rows):
        if i == 0 or row.get('reset_before') or row['segment'] != rows[i-1]['segment']:
            age = 0
        result.append(dict(row, reset_before=(age == 0)))
        age = (age + 1) % 5
    return result


def paired_interval(records, reference, observed, reference_ap):
    def group(rows):
        result = {}
        for r in rows:
            result.setdefault(r['scene'], []).append(r['ap70'])
        return [(np.array([s for r in items for s in r['score']]),
                 np.array([v for r in items for v in r['tp']]).astype(bool),
                 sum(r['gt'] for r in items)) for items in result.values()]

    def ap(groups, draw):
        scores = np.concatenate([groups[i][0] for i in draw])
        tp = np.concatenate([groups[i][1] for i in draw])[np.argsort(-scores)]
        precision = np.cumsum(tp) / np.arange(1,len(tp)+1)
        return np.maximum.accumulate(precision[::-1])[::-1][tp].sum()/sum(groups[i][2] for i in draw)

    def identity(r):
        return r['scene'], r['frame'], r['weak'], r['ap70']['gt']
    assert list(map(identity,records)) == list(map(identity,reference))
    a,b = group(records),group(reference)
    assert len(a)==len(b)==12
    assert abs(ap(a,range(12))-observed)<1e-8 and abs(ap(b,range(12))-reference_ap)<1e-8
    draws = np.random.RandomState(303).randint(12,size=(2000,12))
    return np.percentile([100*(ap(a,d)-ap(b,d)) for d in draws],[2.5,97.5]).tolist()


def evaluate(a, rows, cfg, source, net, criterion, epoch, name='internal', clear=False, window_mode=None):
    report_path = a.output/(name+'.json')
    if report_path.exists():
        return
    frame_log = a.output/(name+'_frames.tmp')
    frame_log.write_text('')
    ledger = ConvLedger(net)
    fit.delta_conv = ledger.count
    net.refreshes = 0
    net.warp_values = 0
    try:
        result = fit.replay(periodic(rows),cfg,source,net,criterion,None,epoch,a.output,{},
                            frame_log=frame_log,memoryless=clear,window_mode=window_mode)
    finally:
        fit.delta_conv = ledger.original
    assert result['scans']==636 and result['labels']==457 and result['weak_total']==924
    n = len(rows)
    ordinary = n - net.refreshes
    costs = []
    for totals in ledger.totals.values():
        dense, mac, ac, mixed = totals.tolist()
        costs.append(dict(event_AC=ac/n, weighted_event_MAC=mac/n, mixed_equal_ops=mixed/n,
                          refresh_dense_MAC=dense/ordinary*net.refreshes/n))
    def load_rows(path):
        return [json.loads(x) for x in path.read_text().splitlines()]
    ref = fit.read(a.reference/'report.json')['ap70']
    ci = paired_interval(load_rows(frame_log), load_rows(a.reference/'frames.jsonl'),result['ap70'],ref)
    os.replace(frame_log,a.output/(name+'_frames.jsonl'))
    result.update(refreshes=net.refreshes,per_scan_residual_stages=costs,checkpoint=fit.digest(a.output/'best.pt'),
                  epoch=epoch,reference_ap70=ref,delta_pp=(result['ap70']-ref)*100,paired_scene_ci95_pp=ci)
    if window_mode:
        result.update(clear_each_frame=clear,ordinary_step_multiplier=net.ordinary_multiplier,
                      bilinear_output_values_per_scan=net.warp_values/n)
        result['ordinary_sender_multipliers'] = ([1.,1.] if net.keep_fine_precision
                                                  else [net.ordinary_multiplier]*2) + [net.ordinary_multiplier]*2
    temp = report_path.with_suffix('.tmp')
    fit.write_json(temp,result)
    os.replace(temp,report_path)


def run(a):
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = fit.load_yaml(str(a.inputs/'config.yaml'))
    cfg['data_augment'] = []
    plan, receipt = fit.read(a.plan),fit.read(a.plan.parent/'report.json')
    assert fit.digest(a.plan)==receipt['plan']
    assert fit.digest(a.initial)=='57f9d146f2a4dd1344ad36dd670bc7f9bcdfd85ae04aaa8c4855cc635b148c33'
    scenes = receipt['splits']['fit']['scenes']
    train = [r for r in plan['rows']['fit'] if r['scene'] in scenes[:-2]]
    dev = periodic([r for r in plan['rows']['fit'] if r['scene'] in scenes[-2:]])
    internal = plan['rows']['internal']
    assert not set(scenes).intersection(r['scene'] for r in internal)
    for r in train+dev+internal:
        assert r['split']=='train'
        for key in ('pcd','label'):
            if r[key]:
                assert fit.digest(Path(r[key])) == plan['members'][r[key]]
    source = fit.build_source(a.inputs,cfg)
    net = RecoveryPyramid(source.core,a.initial,a.variant,cfg['preprocess']['cav_lidar_range']).cuda()
    window_mode = {'coarse_fixed':'fixed','coarse_rigid':'rigid',
                   'fine_temporal':'fixed','fine_independent':'fixed'}.get(a.variant)
    independent = a.variant == 'fine_independent'
    scale_comparison = a.variant in ('fine_temporal','fine_independent')
    heads = list(net.heads.parameters())
    backbone = [p for k,p in net.named_parameters() if not k.startswith('heads.')]
    opt = torch.optim.AdamW([{'params':backbone},{'params':heads}], weight_decay=.0001)
    criterion = fit.PointPillarDirLoss(cfg['loss']['args']).cuda()
    a.output.mkdir(parents=True,exist_ok=True)
    contract = dict(variant=a.variant, seed=a.seed, epochs=a.epochs, stage_caps=[32,16,16],
                    input_cap=32, initial_step_multiplier=.25, reset_interval=5,
                    initial=fit.digest(a.initial), plan=receipt['plan'], training=scenes[:-2], monitor=scenes[-2:],
                    code=fit.digest(Path(__file__)), replay=fit.digest(Path(fit.__file__)),
                    reference=fit.digest(a.reference/'frames.jsonl'),
                    policy='best trained epoch by monitor AP70, earliest tie; fixed cosine schedule',
                    trainable=sum(p.numel() for p in net.parameters() if p.requires_grad))
    cp = a.output/'contract.json'
    if window_mode:
        contract.update(window_mode=window_mode,ordinary_step_multiplier=2.,period=5,
                        pose='causal translation and yaw to first scan' if window_mode=='rigid' else 'identity')
    if scale_comparison:
        contract.update(ordinary_sender_multipliers=[1,1,2,2],independent=independent,
                        lr_horizon=24,optimizer_steps_per_epoch=174)
    if cp.exists():
        assert fit.read(cp)==contract
    else:
        fit.write_json(cp,contract)
    latest = a.output/'latest.pt'
    best, start = -1., 0
    if latest.exists():
        saved = torch.load(latest,map_location='cuda')
        net.load_state_dict(saved['net'])
        opt.load_state_dict(saved['optimizer'])
        best, start = saved['best'],saved['epoch']+1
        del saved
    attempt = a.output/('attempt_'+str(time.time_ns()))
    attempt.mkdir()
    fit.write_json(attempt/'start.json',dict(first_epoch=start))
    point_bins = {}
    for epoch in range(start,a.epochs+1):
        horizon = 24 if scale_comparison else a.epochs
        lr = 5e-6+(3e-5-5e-6)*(1+math.cos(math.pi*max(0,epoch-1)/(horizon-1)))/2
        for group,scale in zip(opt.param_groups,(1.,.5)):
            group['lr'] = lr*scale
        net.refreshes = 0
        trained = fit.replay(train,cfg,source,net,criterion,opt,epoch,attempt,point_bins,
                             order_seed=a.seed,window_mode=window_mode,memoryless=independent) if epoch else None
        train_refreshes = net.refreshes
        if trained and (trained['updates'] != 174 or not all(0 < r[0] < 1 for r in trained['rates'])):
            raise RuntimeError(('training health',trained))
        net.refreshes = 0
        val = fit.replay(dev,cfg,source,net,criterion,None,epoch,a.output,point_bins,
                         window_mode=window_mode,memoryless=independent)
        if not all(0 < r[0] < 1 for r in val['rates']):
            raise RuntimeError(('validation activity',val))
        if epoch and val['ap70']>best:
            best = val['ap70']
            fit.checkpoint(a.output/'best.pt',net,opt,epoch,best,0)
        entry = dict(epoch=epoch,lr=lr,train=trained,validation=val,best_trained_ap70=best,
                     train_exact_frames=train_refreshes,validation_exact_frames=net.refreshes)
        fit.write_json(a.output/('epoch_%02d.json'%epoch),entry)
        fit.checkpoint(latest,net,opt,epoch,best,0)
        print(json.dumps(entry),flush=True)
    saved = torch.load(a.output/'best.pt',map_location='cpu')
    net.load_state_dict(saved['net'])
    fit.write_json(a.output/'train_complete.json',dict(epoch=saved['epoch'],best=saved['best'],
                   epochs=a.epochs,updates=a.epochs*174))
    evaluate(a,internal,cfg,source,net,criterion,saved['epoch'],window_mode=window_mode,clear=independent)
    if scale_comparison:
        evaluate(a,internal,cfg,source,net,criterion,saved['epoch'],'swap_history',not independent,window_mode)
        net.keep_fine_precision=False
        evaluate(a,internal,cfg,source,net,criterion,saved['epoch'],'uniform_coarse',independent,window_mode)
        net.ordinary_multiplier=1.
        evaluate(a,internal,cfg,source,net,criterion,saved['epoch'],'uniform_fine',independent,window_mode)
    elif window_mode:
        evaluate(a,internal,cfg,source,net,criterion,saved['epoch'],'clear_same_phase',True,window_mode)
        if window_mode=='rigid':
            evaluate(a,internal,cfg,source,net,criterion,saved['epoch'],'without_alignment',window_mode='fixed')
        net.ordinary_multiplier=1.
        evaluate(a,internal,cfg,source,net,criterion,saved['epoch'],'fine_all_frames',window_mode=window_mode)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','inputs','initial','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['native','refresh','coarse_fixed','coarse_rigid',
                                       'fine_temporal','fine_independent'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=24)
    run(p.parse_args())
