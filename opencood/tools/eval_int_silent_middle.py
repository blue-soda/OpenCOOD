"""Skip actual middle convolution jobs after receiving a silent fine neighborhood."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate, conv_costs, supports
from opencood.models.sub_modules.int_branch_fusion import BranchFusion
from opencood.models.sub_modules.int_grouped_receiver import GroupedEdge
from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionCache
from opencood.models.sub_modules.int_silent_receiver import PairedEdge, EdgeView
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_deep_block_online import CachedResidualPolicy, ObservedPolicy
from opencood.tools.eval_int_online_half import build
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.eval_int_fusion_cache import complete_scenes
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import compact_group_bucket, spatial_input
from opencood.tools.run_int_silent_receiver import training_rows
from opencood.tools.run_int_ego import loader, to_device, write_json


def support_counts(active):
    return {k:int(v.sum()) for k,v in zip(('output','radius1','radius2'),supports(active))}


class SilencePolicy:
    def __init__(self, receiver, mode):
        self.receiver = receiver
        self.mode = mode

    def __call__(self, previous_skip, valid, coverage, innovation, frame):
        # Each middle output depends on two middle 3x3 rings, followed backward
        # by a stride-two, 3x3 fine observation (eleven fine cells per axis).
        center = F.max_pool2d(self.receiver.event_sites.float(), 3, 2, 1)
        footprint = F.max_pool2d(center, 5, 1, 2)
        eligible = ~previous_skip & valid.bool() & coverage
        silent = eligible & (footprint == 0)
        self.base = ~previous_skip
        self.extra = torch.zeros_like(silent)
        if self.mode == 'silent':
            self.extra = silent
        elif self.mode == 'continuous':
            # Matched-count counterfactual using the existing dense innovation.
            ids = eligible.flatten().nonzero().flatten()
            values = F.max_pool2d(innovation, 5, 1, 2).flatten()[ids]
            tie = ((ids+int(frame))*2654435761) % (1<<32)
            order = np.lexsort((tie.cpu().numpy(), values.cpu().numpy()))
            chosen = torch.as_tensor(order[:int(silent.sum())], device=ids.device)
            self.extra.view(-1)[ids[chosen]] = True
        assert not bool((self.extra & ~eligible).any())
        self.stats = dict(base=support_counts(self.base),eligible=int(eligible.sum()),
                          center_silent=int((eligible & (center==0)).sum()),full_silent=int(silent.sum()),
                          extra=int(self.extra.sum()),overlap_silent=int((self.extra&silent).sum()))
        return previous_skip | self.extra


class InspectMiddle(FineInnovationUpdate):
    def __call__(self, *args):
        self.result = super().__call__(*args)
        return self.result


@torch.no_grad()
def run(args):
    base = args.long_base
    args.inputs = base/'inputs'
    book_path = base/'book_train/report.json'
    book = read(book_path)
    frozen = read(args.calibration/'report.json')
    assert frozen['status']=='COMPLETED' and frozen['threshold']==12
    source, cfg, short, _, _, _, q = prepare_stream(args.inputs, base/'clips')
    data_rows, traces, identity = (training_rows(base,book_path,True) if args.audit
                                  else complete_scenes(args,short))
    if args.grouped:
        paired = GroupedEdge(source.bounds,16384,book['levels'],charge=args.charge)
        routes = dict(point=('point','silent'),tile=('tile','silent'),reset=('reset','silent'),
                      fp=('float_bytes','silent'),tile_base=('tile','base'))
        if args.charge:
            routes = dict(tile=('tile','silent'),charge=('charge','silent'),fp=('float_bytes','silent'),
                          fp_charge=('float_bytes_charge','silent'),charge_base=('charge','base'))
        elif args.controls:
            routes = dict(point_base=('point','base'),tile_score=('tile','continuous'))
    else:
        paired = PairedEdge(source.bounds, book['cap']//2, book['levels'], [12], calibration=True)
        routes = {n:('12',n) for n in ('base','silent','continuous')}
    for receiver in paired.arms.values():
        receiver.capture_sites = True
    core = source.core
    fusion = core.readout.layers[0].double_conv[0]
    layer = (BranchFusion(fusion[0],verify=False) if args.grouped else NonzeroFusionCache(fusion[0])).eval()
    fusion[0] = layer
    names = tuple(routes)
    models, policies = {}, {}
    for n in names:
        key,mode = routes[n]
        view = EdgeView(paired,key)
        model = build(source,q,view)
        gate = SilencePolicy(paired.arms[key],mode)
        model.coarse_executor = InspectMiddle(False,model.coarse_executor.receive_fine,gate)
        models[n],policies[n] = model,gate
    costs = conv_costs(core)
    data, stream = loader(data_rows,cfg,0,303)
    write_json(args.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        data=identity,scenes=sorted({r['scene'] for r in data_rows}),calibration=digest(args.calibration/'report.json'),
        threshold=12,cap=paired.cap,book=digest(book_path),cost_per_site=costs,training=False,audit=args.audit,
        grouped=args.grouped,charge=args.charge,routes=routes,params=sum(p.numel() for p in source.parameters())))
    labelled=ordinary=0
    with (args.output/'frames.jsonl').open('x',buffering=1) as log:
        for i,(sample,trace) in enumerate(zip(stream,traces)):
            meta=sample['meta']
            assert meta['frame']==trace['frame']
            if i==0 or meta['segment']!=data_rows[i-1]['segment']:
                states=dict.fromkeys(names,None if trace['refresh']=='initial' else dict(scene=''))
                selectors={n:ObservedPolicy(models[n],partial(CachedResidualPolicy,compiler=compact_group_bucket)) for n in names}
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            reason=trace['refresh']
            paired.start(meta,spatial.shape[-2:],reason)
            gt=data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            labelled += int(meta['supervised'])
            ordinary += int(reason is None)
            row=dict(frame=meta['frame'],scene=meta['scene'],segment=meta['segment'],refresh=reason,arms={})
            for n,model in models.items():
                prev,selector=states[n],selectors[n]
                assert model.refresh_reason(prev,meta)==reason
                model.coarse_projector.begin(prev,reason)
                if args.grouped:
                    layer.begin(prev,reason)
                else:
                    layer.begin(prev,reason,True)
                selector.observation=None
                (pred,state,info),cost=measured_step(model,spatial,prev,meta,selector)
                assert reason==info['full_refresh']
                rec=dict(core=cost+layer.ledger['float_conv'])
                if args.grouped:
                    rec['fusion']=layer.ledger['float_conv']
                if reason is None:
                    exe,gate=model.coarse_executor,policies[n]
                    rec.update(gate.stats,executed=exe.counts,middle=sum(costs[k]*v for k,v in exe.counts.items()))
                    rec['core']+=rec['middle']
                    assert exe.counts==support_counts(~exe.skip)
                    if args.audit:
                        fine,old,valid,held=exe.inputs
                        full=core.coarse_update(core.coarse_light(fine),old,valid)
                        ref=torch.where(exe.skip,held,full)
                        err=(ref-exe.result).abs()
                        assert torch.allclose(ref,exe.result,atol=1e-4,rtol=1e-5),(meta['frame'],float(err.max()))
                        assert torch.equal(exe.result.masked_select(exe.skip),held.masked_select(exe.skip))
                        delta=(full-held)[0,:,gate.extra[0,0]].abs()
                        rec.update(max_error=float(err.max()),hold_l1=float(delta.sum(dtype=torch.float64)),
                                   hold_cells=delta.numel())
                    exe.inputs=exe.result=None
                state['branch_fusion' if args.grouped else 'fusion_cache']=layer.new
                states[n]=state
                if gt is not None: rec.update(score(data.post,ego,pred,gt,record_matches=True))
                row['arms'][n]=rec
            row['wire']=paired.stats if args.grouped else paired.stats['12']
            if reason is None and not args.grouped:
                a,b,c=(row['arms'][n] for n in names)
                assert a['base']==b['base']==c['base'] and b['extra']==c['extra']==b['full_silent']
            log.write(json.dumps(row)+'\n')
    assert i+1==len(data_rows)==(211 if args.audit else 1014)
    assert labelled==(0 if args.audit else 458)
    write_json(args.output/'report.json',dict(status='COMPLETED',scans=len(data_rows),labels=labelled,
        ordinary=ordinary,optimizer_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--audit',action='store_true')
    p.add_argument('--grouped',action='store_true')
    p.add_argument('--charge',action='store_true')
    p.add_argument('--controls',action='store_true')
    for k in ('long-base','calibration','vehicle','output'):
        p.add_argument('--'+k,type=Path,required=k!='vehicle')
    a=p.parse_args()
    if a.charge and not a.grouped:p.error('spatial charge requires grouped routing')
    if a.controls and (not a.grouped or a.charge):p.error('controls require the uncharged grouped sender')
    if not a.audit and a.vehicle is None:p.error('development evaluation needs vehicle directory')
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)
