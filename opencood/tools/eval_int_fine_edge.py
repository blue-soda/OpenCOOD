"""Frozen fine-edge quality intervention and independent dense-receiver replay."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate, compact_update, conv_costs
from opencood.models.sub_modules.int_cross_scale_dw import CrossScaleCache
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_fine_edge import FineEdgeSender
from opencood.models.sub_modules.int_fine_demand import QuotaFineEdge
from opencood.models.sub_modules.int_fine_codebook import FineCodebook
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


class MatchedMiddle:
    def __call__(self, core, fine, old, valid, coverage, state, meta, dt, spatial_shape, bounds):
        assert not bool((self.skip & ~coverage).any())
        output,self.counts = compact_update(core,fine,old,valid,dt,~self.skip,old)
        return output


@torch.no_grad()
def run(args):
    source,cfg,rows,traces,starts,members,q = prepare_stream(args.inputs,args.scans)
    baseline = [json.loads(line) for line in (args.prior/'frames.jsonl').read_bytes().splitlines()]
    assert digest(args.prior/'frames.jsonl')==read(args.reference/'contract.json')['reference']
    checks = [json.loads(line) for line in (args.reference/'frames.jsonl').read_bytes().splitlines()]
    assert members==read(args.prior/'contract.json')['members']
    modes = ('continuous','carry','dense') if args.engineering else ('continuous','carry','drop')
    if args.phase_test:
        modes = ('continuous','ordinary','refresh')
    cap=None
    if args.quota_source:
        assert digest(args.quota_source)=='033356e8202a519b63db092e35f9f77fa232c31ba6b2abc8346a0f873412db76'
        train=[json.loads(line) for line in args.quota_source.read_bytes().splitlines()]
        assert not {t['scene'] for t in train}.intersection(r['scene'] for r in rows)
        cap=min(t['arms']['deferred']['fine']['addresses'] for t in train if t['refresh'] is None)//2
        modes=('continuous','unit_carry','unit_round','amplitude')
    calibration=None
    if args.book:
        calibration=read(args.book)
        assert calibration['status']=='COMPLETED' and calibration['cap']==cap
        assert not set(calibration['scenes']).intersection(r['scene'] for r in rows)
        modes=('continuous','power','power_ann','calibrated')
    models = {}
    for mode in modes:
        phase = mode if mode in ('ordinary','refresh') else 'all'
        sender = None if mode=='continuous' else FineEdgeSender(source.bounds,dense=mode=='dense',send_on=phase)
        if cap is not None and mode!='continuous':
            if calibration is None:
                sender=QuotaFineEdge(source.bounds,mode,cap)
            else:
                table=calibration['levels'] if mode=='calibrated' else [1,2,4,8]
                sender=FineCodebook(source.bounds,cap,table,analog=mode=='power_ann')
        model = CrossScaleCache(source,'sparse_amplitude',planner=plan_host_coalesced_refresh,
                                deep_projector=DeepSpikeProjection(source.core.deblocks[2],q,'amplitude'),
                                fine_to_coarse=sender)
        model.coarse_interface=model.coarse_projector=MiddleDecayProjection(source.core.deblocks[1],'zero','amplitude')
        model.coarse_executor=FineInnovationUpdate(False) if sender is None else MatchedMiddle()
        models[mode]=model
    write_json(args.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               params=sum(p.numel() for p in source.parameters()), reference=digest(args.reference/'frames.jsonl'),
               reference_masks_from='simultaneous continuous causal replay',modes=modes,seed=303,q=.25,cap=cap,
               book=digest(args.book) if args.book else None,
               trace=digest(args.inputs/'traces.json'),weights={k:digest(args.inputs/k) for k in
               ('parent.pth','protected_light_final.pth','readout.pth','quantum.json')},device=torch.cuda.get_device_name()))
    data,stream=loader(rows,cfg,0,303)
    costs=conv_costs(source.core)
    n=0
    with (args.output/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace,old,check in zip(stream,traces,baseline,checks):
            meta=sample['meta']
            reason=trace['refresh']
            frame=meta['frame']
            assert frame==trace['frame']==old['frame']==check['frame']
            if frame in starts:
                states=dict.fromkeys(modes)
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            request=local_request(trace_masks(trace,spatial.device))
            gt=data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] and not args.engineering else None
            row=dict(frame=frame,scene=meta['scene'],refresh=reason,arms={})
            for name,model in models.items():
                prev=states[name]
                sender=model.fine_to_coarse
                assert model.refresh_reason(prev,meta)==reason
                model.coarse_projector.begin(prev,reason)
                if sender is not None:
                    sender.begin(prev,meta,spatial.shape[-2:],reason)
                    if reason is None:
                        model.coarse_executor.skip=mask
                        if cap is not None:
                            sender.active=~mask
                (pred,state,info),core=measured_step(model,spatial,prev,meta,request)
                if reason is None:
                    exe=model.coarse_executor
                    if name=='continuous':
                        mask=exe.skip
                        exe.inputs=None
                    assert exe.counts==old['arms']['zero_zero']['support']
                    core+=sum(costs[k]*v for k,v in exe.counts.items())
                assert core==check['arms']['dense']['core_float_conv']
                entry=dict(core=core,edge=info.get('edge',{}),middle=model.coarse_projector.ledger,deep=model.deep_projector.ledger)
                if sender is not None:
                    sender.commit(state,name=='drop')
                    entry['fine']=sender.ledger
                if name in ('carry','ordinary','refresh'):
                    ref_pred=pred
                    assert torch.equal(states['continuous']['value'],state['value'])
                if name=='dense':
                    assert all(torch.equal(pred[k],ref_pred[k]) for k in ref_pred)
                    for k in ('value','coarse','deep','fine_levels','coarse_projection','deep_projection'):
                        assert torch.equal(state[k],states['carry'][k]), (frame,k)
                    if reason is None:
                        assert all(torch.equal(a,b) for a,b in zip(state['edge_cache'],states['carry']['edge_cache']))
                if name=='power':
                    ref_pred=pred
                if name=='power_ann':
                    assert all(torch.equal(pred[k],ref_pred[k]) for k in pred)
                    for k in ('value','coarse','deep','fine_levels','coarse_projection','deep_projection'):
                        assert torch.equal(state[k],states['power'][k]), (frame,k)
                if gt is not None:
                    # Normalize threshold keys before comparing the archived JSON scores.
                    entry.update(json.loads(json.dumps(score(data.post,ego,pred,gt))))
                    if name=='continuous':
                        assert all(entry[k]==check[k] for k in ('ap','targets','hits')),frame
                states[name]=state
                row['arms'][name]=entry
            log.write(json.dumps(row)+'\n')
            n+=1
    assert n==len(rows)==len(baseline)==len(checks)==272
    write_json(args.output/'report.json',dict(status='COMPLETED',scans=n,modes=modes,matched_masks=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inputs','scans','prior','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--engineering',action='store_true')
    p.add_argument('--phase-test',action='store_true')
    p.add_argument('--quota-source',type=Path)
    p.add_argument('--book',type=Path)
    args=p.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    run(args)

