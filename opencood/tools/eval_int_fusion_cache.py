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
from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionCache
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


def complete_scenes(a, short_rows, remainder=False):
    """Recover complete segments from the six development scenes or their complement."""
    manifest=a.long_base/'r328_manifest.json'
    plan_path=a.long_base/'r309_full_plan.json'
    trace_path=a.long_base/'r309_full_frames.jsonl'
    hashes={p.name:digest(p) for p in (manifest,plan_path,trace_path)}
    assert hashes[plan_path.name]=='1be45bac9fe7301f91616fa4a7bc2c9c14447ac4053421338d4680ac259e82e6'
    assert hashes[trace_path.name]=='adb39a232c147201ecfb708b165742a8a22717b9ad831308a6e868ed5e80d982'
    plan=read(plan_path)
    assert hashes[manifest.name]==plan['manifest_sha256'] and digest(a.inputs/'config.yaml')==plan['config_sha256']
    native=read(manifest)['val']
    frozen=[json.loads(line) for line in trace_path.read_bytes().splitlines()]
    assert len(native)==len(frozen)==len(plan['ordered_stream'])==5794
    scenes={r['scene'] for r in short_rows}
    rows,traces=[],[]
    for r,t,identity in zip(native,frozen,plan['ordered_stream']):
        if (r['scene'] in scenes)==remainder:continue
        assert all(r[k]==v for k,v in identity.items())
        assert all(r[k]==t[k] for k in ('frame','scene','segment','supervised'))
        for key,folder in (('pcd','velodyne'),('label','label/lidar')):
            if r[key] is None:continue
            path=a.vehicle/folder/Path(r[key]).name
            receipt=plan['input_files'][r[key]]
            assert path.stat().st_size==receipt['bytes'] and digest(path)==receipt['sha256'],path
            r[key]=str(path)
        rows.append(r)
        traces.append(t)
    if remainder:
        assert len(rows)==4780 and not any(r['scene'] in scenes for r in rows)
    else:
        assert len(rows)==1014 and len({r['segment'] for r in rows})==34
        assert {r['frame'] for r in short_rows} <= {r['frame'] for r in rows}
    return rows,traces,hashes


def replay_written(trace):
    def request(frame,mandatory):
        written=torch.zeros_like(mandatory)
        written.view(-1)[trace['written_indices']]=True
        assert int(written.sum())==trace['written'] and not bool((mandatory & ~written).any())
        assert int((~frame.valid).sum())==trace['invalid']
        return written
    return request if trace['refresh'] is None else None


@torch.no_grad()
def run(a):
    source,cfg,rows,traces,starts,_,q=prepare_stream(a.inputs,a.scans)
    book=read(a.book)
    assert book['levels']==[3,6,13,28] and book['cap']==428149
    if a.long_base:
        rows,traces,hashes=complete_scenes(a,rows)
        starts={r['frame'] for i,r in enumerate(rows) if i==0 or rows[i-1]['segment']!=r['segment']}
        old=[None]*len(rows)
        protocol=dict(native=hashes,scope='complete six development scenes, historical R309 written masks')
    else:
        old=[json.loads(x) for x in a.prior.read_bytes().splitlines()]
        protocol=dict(prior=digest(a.prior))
    core=source.core
    base=core.readout.layers[0].double_conv[0]
    layer=(NonzeroFusionCache(base[0]) if a.nonzero else ExactFusionCache(base[0])).eval()
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
               params=sum(p.numel() for p in source.parameters()),block=layer.block,book=digest(a.book),**protocol,
               trace=digest(a.inputs/'traces.json'),amplitude=a.amplitude,nonzero=a.nonzero,optimizer_steps=0))
    n=0
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace,ref in zip(stream,traces,old):
            meta=sample['meta']
            frame,reason=meta['frame'],trace['refresh']
            assert frame==trace['frame'] and (ref is None or ref['frame']==frame)
            if frame in starts:
                states=dict.fromkeys(models,dict(scene='') if a.long_base and reason!='initial' else None)
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            req=replay_written(trace) if a.long_base else local_request(trace_masks(trace,spatial.device))
            row=dict(frame=frame,scene=meta['scene'],segment=meta['segment'],refresh=reason,arms={})
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
                if ref is not None:
                    original=ref['arms']['amplitude' if a.amplitude else 'calibrated']
                    assert sender.ledger==original['fine']
                    assert other+2*384*128*9*layer.ledger['sites']==original['core']
                elif reason is None:
                    actual=(state['deep_age']==0).flatten().nonzero().flatten().tolist()
                    assert actual==trace['written_indices']
                state['fusion_cache']=layer.new
                row['arms'][name]=dict(layer.ledger,core=other+layer.ledger['float_conv'])
                if name=='dense':
                    reference=pred
                    row['frame_age']=state['since_refresh']
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
    assert n==len(rows)==len(old)==(1014 if a.long_base else 272)
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=n,bitwise_equal=True))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','scans','book','output'):p.add_argument('--'+k,type=Path,required=True)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--prior',type=Path)
    src.add_argument('--long-base',type=Path)
    p.add_argument('--vehicle',type=Path)
    p.add_argument('--amplitude',action='store_true')
    p.add_argument('--nonzero',action='store_true')
    a=p.parse_args()
    if a.long_base and not a.vehicle:p.error('--long-base requires --vehicle')
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)

