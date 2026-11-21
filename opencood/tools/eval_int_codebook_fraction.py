"""Paired fine task-memory intervention at fixed middle jobs and deep write opportunities."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate, conv_costs
from opencood.models.sub_modules.int_cross_scale_dw import CrossScaleCache
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_fine_codebook import AdjacentFineCodebook,FineCodebook
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionCache
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_fine_edge import MatchedMiddle
from opencood.tools.eval_int_fusion_cache import complete_scenes, replay_written
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json


@torch.no_grad()
def run(args):
    source,cfg,short,_,_,_,q=prepare_stream(args.inputs,args.scans)
    rows,traces,hashes=complete_scenes(args,short)
    prior=[json.loads(line) for line in args.prior.read_bytes().splitlines()]
    assert len(prior)==len(rows) and read(args.prior.parent/'report.json')['bitwise_equal']
    book=read(args.book)
    assert book['levels']==[3,6,13,28] and book['cap']==428149
    core=source.core
    fusion=core.readout.layers[0].double_conv[0]
    layer=NonzeroFusionCache(fusion[0]).eval()
    fusion[0]=layer
    models={}
    modes=('continuous',) if args.continuous else ('carry','adjacent' if args.adjacent else 'drop')
    support={r['frame']:r for r in map(json.loads,args.support.read_bytes().splitlines())} if args.support else {}
    if args.support:
        assert len(support)==sum(r['supervised'] for r in rows)
    for name in modes:
        edge_type=AdjacentFineCodebook if name=='adjacent' else FineCodebook
        edge=None if name=='continuous' else edge_type(source.bounds,book['cap'],book['levels'])
        def receive(value,active,sender=edge):
            sender.active=active
            return sender(value)
        # Carry computes the middle mask from its own fine state. Drop receives
        # that same mask before encoding its own events for the middle stage.
        transform=(lambda value,s=edge:s(value) if s.phase=='refresh' else value) if name=='carry' else edge
        model=CrossScaleCache(source,'sparse_amplitude',planner=plan_host_coalesced_refresh,
                              deep_projector=DeepSpikeProjection(core.deblocks[2],q,'amplitude'),fine_to_coarse=transform)
        model.coarse_interface=model.coarse_projector=MiddleDecayProjection(core.deblocks[1],'zero','amplitude')
        model.coarse_executor=MatchedMiddle() if name in ('drop','adjacent') else FineInnovationUpdate(False,receive if edge else None)
        models[name]=(model,edge)
    per=conv_costs(core)
    data,stream=loader(rows,cfg,0,303)
    write_json(args.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               native=hashes,prior=digest(args.prior),book=digest(args.book),optimizer_steps=0,seed=303,
               params=sum(p.numel() for p in source.parameters()),quantum=.25,modes=modes,
               support=digest(args.support) if args.support else None,
               intervention=('continuous fine-to-middle' if args.continuous else 'adjacent task difference' if args.adjacent
                             else 'ordinary after readout, round fine task on read area')))
    with (args.output/'frames.jsonl').open('x',buffering=1) as log:
        for i,(sample,trace,ref) in enumerate(zip(stream,traces,prior)):
            meta=sample['meta']
            frame,reason=meta['frame'],trace['refresh']
            assert frame==trace['frame']==ref['frame'] and reason==ref['refresh']
            if i==0 or meta['segment']!=rows[i-1]['segment']:
                states=dict.fromkeys(models,dict(scene='') if reason!='initial' else None)
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            gt=data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            row=dict(frame=frame,scene=meta['scene'],segment=meta['segment'],refresh=reason,frame_age=ref['frame_age'],arms={})
            if args.support and gt is not None:
                cpu_gt=data.post.generate_gt_bbx(dict(ego=sample['ego']))
                receipt=support[frame]
                assert hashlib.sha256(cpu_gt.numpy().tobytes()).hexdigest()==receipt['gt_xyz_sha256']
                assert torch.allclose(cpu_gt,gt.cpu(),rtol=0,atol=1e-4)
                assert len(receipt['points'])==len(gt)
            for name,(model,edge) in models.items():
                prev=states[name]
                assert model.refresh_reason(prev,meta)==reason
                if edge is not None:edge.begin(prev,meta,spatial.shape[-2:],reason)
                model.coarse_projector.begin(prev,reason)
                layer.begin(prev,reason,True)
                if name in ('drop','adjacent') and reason is None:
                    model.coarse_executor.skip=mask
                    edge.active=~mask
                (pred,state,info),other=measured_step(model,spatial,prev,meta,replay_written(trace))
                stats={}
                if reason is None:
                    exe=model.coarse_executor
                    other+=sum(per[k]*v for k,v in exe.counts.items())
                    if name in ('carry','continuous'):
                        mask=exe.skip
                        exe.inputs=None
                        if args.support:
                            row['middle_mask_sha256']=hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
                    elif name=='drop':
                        assert exe.counts==models['carry'][0].coarse_executor.counts
                        task=state['value']
                        rounded=(task/edge.quantum).round()*edge.quantum
                        revised=torch.where(edge.read,rounded,task)
                        diff=(task-revised).abs()
                        stats.update(changed=int(diff.count_nonzero()),max_delta=float(diff.max()),sum_delta=float(diff.sum()))
                        assert stats['max_delta']<=edge.quantum/2
                        state['value']=revised
                    assert (state['deep_age']==0).flatten().nonzero().flatten().tolist()==trace['written_indices']
                    if edge is not None:
                        stats.update(addresses=edge.ledger['addresses'],bytes=edge.ledger['bytes'],units=edge.ledger['units'])
                if edge is not None:edge.commit(state,False)
                if name=='adjacent':
                    assert torch.equal(state['value'],states['carry']['value'])
                    if reason is None:
                        assert model.coarse_executor.counts==models['carry'][0].coarse_executor.counts
                        assert stats['addresses']==row['arms']['carry']['addresses'] and stats['bytes']==row['arms']['carry']['bytes']
                state['fusion_cache']=layer.new
                stats['core']=other+layer.ledger['float_conv']
                if gt is not None:stats.update(score(data.post,ego,pred,gt,record_matches=bool(args.support)))
                if name=='carry':
                    assert stats['core']==ref['arms']['local']['core']
                    if gt is not None:
                        assert {str(t):v for t,v in stats['ap'].items()}==ref['ap']
                        assert stats['hits']==ref['hits'] and stats['targets']==ref['targets']
                row['arms'][name]=stats
                states[name]=state
            log.write(json.dumps(row)+'\n')
    assert i+1==len(rows)
    write_json(args.output/'report.json',dict(status='COMPLETED',scans=len(rows),carry_prior_exact='carry' in models))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('inputs','scans','book','prior','vehicle','long-base','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    control=parser.add_mutually_exclusive_group()
    control.add_argument('--continuous',action='store_true')
    control.add_argument('--adjacent',action='store_true')
    parser.add_argument('--support',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args)

