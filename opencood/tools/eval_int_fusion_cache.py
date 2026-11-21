"""Paired closed-loop detection with an actual sparse deep fusion contribution."""
import argparse
from pathlib import Path
import json
import subprocess
import torch

from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate,conv_costs
from opencood.models.sub_modules.int_cross_scale_dw import CrossScaleCache
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_exact_fusion_cache import ExactFusionCache
from opencood.models.sub_modules.int_fine_codebook import FineCodebook
from opencood.models.sub_modules.int_fine_demand import QuotaFineEdge
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_spike_stream import prepare_stream,read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_deep_block_adapt import local_request,trace_masks


@torch.no_grad()
def run(a):
    source,cfg,rows,traces,starts,_,q=prepare_stream(a.inputs,a.scans)
    book=read(a.book)
    assert book['levels']==[3,6,13,28] and book['cap']==428149
    old=[json.loads(x) for x in a.prior.read_bytes().splitlines()]
    core=source.core
    base=core.readout.layers[0].double_conv[0]
    layer=ExactFusionCache(base[0]).eval()
    base[0]=layer
    models={}
    for name in ('dense','local'):
        sender=(QuotaFineEdge(source.bounds,'amplitude',book['cap']) if a.amplitude
                else FineCodebook(source.bounds,book['cap'],book['levels']))
        def receive(fine,active,s=sender):
            s.active=active
            return s(fine)
        model=CrossScaleCache(source,'sparse_amplitude',planner=plan_host_coalesced_refresh,
                              deep_projector=DeepSpikeProjection(core.deblocks[2],q,'amplitude'),
                              fine_to_coarse=lambda v,s=sender:s(v) if s.phase=='refresh' else v)
        model.coarse_interface=model.coarse_projector=MiddleDecayProjection(core.deblocks[1],'zero','amplitude')
        model.coarse_executor=FineInnovationUpdate(False,receive_fine=receive)
        models[name]=(model,sender)
    data,stream=loader(rows,cfg,0,303)
    per=conv_costs(core)
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               params=sum(p.numel() for p in source.parameters()),block=layer.block,book=digest(a.book),prior=digest(a.prior),
               trace=digest(a.inputs/'traces.json'),amplitude=a.amplitude,optimizer_steps=0))
    n=0
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace,ref in zip(stream,traces,old):
            meta=sample['meta']
            frame,reason=meta['frame'],trace['refresh']
            assert frame==trace['frame']==ref['frame']
            if frame in starts:states=dict.fromkeys(models)
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            req=local_request(trace_masks(trace,spatial.device))
            row=dict(frame=frame,scene=meta['scene'],refresh=reason,arms={})
            for name,(model,sender) in models.items():
                prev=states[name]
                assert model.refresh_reason(prev,meta)==reason
                sender.begin(prev,meta,spatial.shape[-2:],reason)
                model.coarse_projector.begin(prev,reason)
                layer.begin(prev,reason,name=='local')
                (pred,state,info),other=measured_step(model,spatial,prev,meta,req)
                if reason is None:
                    other+=sum(per[k]*v for k,v in model.coarse_executor.counts.items())
                    model.coarse_executor.inputs=None
                sender.commit(state,False)
                original=ref['arms']['amplitude' if a.amplitude else 'calibrated']
                assert sender.ledger==original['fine']
                assert other+2*384*128*9*layer.ledger['sites']==original['core']
                state['fusion_cache']=layer.new
                row['arms'][name]=dict(layer.ledger,core=other+layer.ledger['float_conv'])
                if name=='dense':
                    reference=pred
                    if meta['supervised']:
                        gt=data.post.generate_gt_bbx(dict(ego=ego))
                        row.update(score(data.post,ego,pred,gt))
                else:
                    assert all(torch.equal(a,b) for a,b in zip(state['fusion_cache'],states['dense']['fusion_cache'])),frame
                    assert all(torch.equal(pred[k],reference[k]) for k in pred),frame
                    assert all(torch.equal(state[k],states['dense'][k]) for k in ('value','coarse','deep','fine_levels'))
                states[name]=state
            log.write(json.dumps(row)+'\n')
            n+=1
    assert n==len(rows)==len(old)==272
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=n,bitwise_equal=True))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','scans','book','prior','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--amplitude',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)

