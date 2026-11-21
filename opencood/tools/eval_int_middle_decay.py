"""Evaluate both consumers of an age-decayed middle BEV on the frozen six clips."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.tools.calibrate_int_middle_budget import detector
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


@torch.no_grad()
def run(inputs, scans, reference, out, unit_prefix, shared_phase):
    source,cfg,rows,traces,starts,members,q=prepare_stream(inputs,scans)
    assert read(reference/'contract.json')['members']==members
    old={r['frame']:r for r in map(json.loads,(reference/'frames.jsonl').read_text().splitlines())}
    arms={'zero':('zero','amplitude'),'rounded':('rounded','amplitude'),
          'factor':('factor','amplitude'),'factor_dense':('factor','dense')}
    if shared_phase:
        arms={'factor':('factor','amplitude'),'phase':('phase','amplitude'),'phase_dense':('phase','dense')}
    if unit_prefix:
        name='phase' if shared_phase else 'factor'
        arms={name:(name,'amplitude'),name+'_unit':(name,'unit')}
        ids={f for clip in read(inputs/'plan.json')['clips'][:2] for f in clip[:12]}
        rows=[r for r in rows if r['frame'] in ids]
        traces=[t for t in traces if t['frame'] in ids]
    models={}
    table_bytes={}
    for name,(predictor,execution) in arms.items():
        model=detector(source,q,'carry',.25)
        receiver=MiddleDecayProjection(source.core.deblocks[1],predictor,execution)
        model.coarse_interface=model.coarse_projector=receiver
        models[name]=model
        table_bytes[name]=sum(t.numel()*t.element_size() for t in receiver.unit_tables.values())
    write_json(out/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        members=members,prior=digest(reference/'frames.jsonl'),params=sum(p.numel() for p in source.parameters()),
        arms=arms,scaled_unit_table_bytes=table_bytes,device=torch.cuda.get_device_name(),torch=torch.__version__,
        quantum=.25,age_gain=[.875,.875,.875,1.],trace=digest(inputs/'traces.json'),
        weights={k:digest(inputs/k) for k in ['parent.pth','protected_light_final.pth','readout.pth','quantum.json']}))
    data,stream=loader(rows,cfg,0,303)
    n=0
    with (out/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace in zip(stream,traces):
            meta=sample['meta']
            assert trace['frame']==meta['frame'] and trace['supervised']==meta['supervised']
            if meta['frame'] in starts:
                states=dict.fromkeys(models)
            preds={}
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            request=local_request(trace_masks(trace,spatial.device))
            gt=data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] and not unit_prefix else None
            row=dict(frame=meta['frame'],scene=meta['scene'],refresh=trace['refresh'],arms={})
            for name,model in models.items():
                receiver=model.coarse_projector
                reason=model.refresh_reason(states[name],meta)
                assert reason==trace['refresh']
                receiver.begin(states[name],reason)
                (pred,state,info),core=measured_step(model,spatial,states[name],meta,request)
                assert info['full_refresh']==reason
                states[name],preds[name]=state,pred
                cache=state['coarse_projection']
                entry=dict(middle=receiver.ledger,deep=model.deep_projector.ledger,core_float_flops=core,
                           cache_bytes=cache.numel()*cache.element_size())
                archived=old[meta['frame']]['arms']['middle_carry']
                assert core+2*receiver.ledger['dense_mac']==archived['core_float_conv_flops']
                if name=='zero' and reason is None:
                    assert receiver.ledger['units']==archived['middle']['units']
                    assert receiver.ledger['nonzero']==archived['middle']['changed_channels']
                if name in ('factor_unit','phase_unit','factor_dense','rounded'):
                    ref='zero' if name=='rounded' else name.split('_')[0]
                    assert all(torch.equal(state[k],states[ref][k]) for k in
                               ('coarse_projection','coarse','deep','deep_projection'))
                    assert all(torch.equal(v,preds[ref][k]) for k,v in pred.items())
                if name=='phase_dense':
                    for k in ('coarse_projection','coarse','deep','deep_projection'):
                        assert torch.equal(state[k],states['phase'][k]), (meta['frame'],k)
                    assert all(torch.equal(v,preds['phase'][k]) for k,v in pred.items())
                if name=='phase' and not unit_prefix:
                    multiplier=7**(3-min(receiver.age,3))
                    assert torch.equal(cache,states['factor']['coarse_projection']*multiplier)
                    assert torch.equal(state['deep'],states['factor']['deep'])
                    assert torch.equal(state['deep_projection'],states['factor']['deep_projection'])
                    assert receiver.ledger['units']==models['factor'].coarse_projector.ledger['units']
                    entry['max_head_abs_vs_factor']={k:float((v-preds['factor'][k]).abs().max())
                                                     for k,v in pred.items()}
                if gt is not None and name in ('zero','factor','phase'):
                    entry.update(score(data.post,ego,pred,gt))
                row['arms'][name]=entry
            log.write(json.dumps(row)+'\n')
            n+=1
    assert n==len(rows)==(24 if unit_prefix else 272)
    write_json(out/'report.json',dict(status='COMPLETED',scans=n,exact_peer_states_and_heads=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('inputs','scans','reference','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--unit-prefix',action='store_true')
    parser.add_argument('--shared-phase',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args.inputs,args.scans,args.reference,args.output,args.unit_prefix,args.shared_phase)
