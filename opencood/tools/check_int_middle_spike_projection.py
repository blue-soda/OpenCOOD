"""Check dense, amplitude and unit middle receivers on fixed causal prefixes."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.tools.calibrate_int_middle_budget import detector
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


@torch.no_grad()
def run(inputs, scans, reference, output):
    source,cfg,rows,traces,starts,members,q=prepare_stream(inputs,scans)
    contract=read(reference/'contract.json')
    assert contract['members']==members and contract['quantum']==.25
    modes=('float','dense','amplitude','unit')
    models={name:detector(source,q,'carry',.25) for name in modes}
    block=source.core.deblocks[1]
    channel_q=torch.full((block[0].in_channels,),.25,device='cuda')
    for name in modes[1:]:
        models[name].coarse_projector=DeepSpikeProjection(block,channel_q,name)
    clips=read(inputs/'plan.json')['clips'][:2]
    ids=[frame for c in clips for frame in c[:12]]
    by_id={t['frame']:(r,t) for r,t in zip(rows,traces)}
    archive={r['frame']:r for r in map(json.loads,(reference/'frames.jsonl').read_text().splitlines())}
    data,stream=loader([by_id[i][0] for i in ids],cfg,0,303)
    write_json(output/'contract.json',dict(source=contract['source'],members=members,
        code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),frames=ids,
        middle_quantum=.25,params=contract['params'],device=torch.cuda.get_device_name(),
        integer='int64 levels and linear sums, output-channel int16 weights, dense rebuild on refresh'))
    with (output/'frames.jsonl').open('x',buffering=1) as raw:
        for frame,sample in zip(ids,stream):
            meta=sample['meta']
            assert meta['frame']==frame
            trace=by_id[frame][1]
            if frame in starts:
                states=dict.fromkeys(modes)
            spatial=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
            mask=trace_masks(trace,spatial.device)
            preds, costs={},{}
            for name,model in models.items():
                (preds[name],states[name],info),costs[name]=measured_step(
                    model,spatial,states[name],meta,local_request(mask))
                assert info['full_refresh']==trace['refresh']
                if name=='float':
                    assert costs[name]==archive[frame]['arms']['middle_carry']['core_float_conv_flops']
                    continue
                state=states[name]
                for k in ('value','coarse','deep','deep_age','deep_projection'):
                    assert torch.equal(state[k], states['float'][k]), (name,k,frame)
                assert torch.equal(state['coarse_projection'],states['dense']['coarse_projection'])
                for k in preds[name]:
                    assert torch.equal(preds[name][k],preds['dense'][k]),(name,k,frame)
            ledger=models['unit'].coarse_projector.ledger
            assert ledger==models['amplitude'].coarse_projector.ledger==models['dense'].coarse_projector.ledger
            shadow=archive[frame]['arms']['middle_carry']['middle']
            assert ledger['units']==shadow['units'] and ledger['nonzero']==shadow['changed_channels']
            assert costs['float']-costs['unit']==2*ledger['dense_mac']
            cache=states['unit']['coarse_projection']
            raw.write(json.dumps(dict(frame=frame,refresh=trace['refresh'],middle=ledger,
                core_float_conv_flops=costs,cache_bytes=cache.numel()*cache.element_size(),
                head_max_abs_vs_float={k:float((v-preds['float'][k]).abs().max()) for k,v in preds['unit'].items()}))+'\n')
    assert len((output/'frames.jsonl').read_text().splitlines())==len(ids)
    write_json(output/'report.json',dict(status='COMPLETED',scans=len(ids),integer_cache_exact=True,heads_exact=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('inputs','scans','reference','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args.inputs,args.scans,args.reference,args.output)
