"""Protected refresh and task-potential event engineering on fixed real inputs."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector,LeakyStateUpdate,reset_reason
from opencood.models.sub_modules.int_task_membrane import TaskPotentialEvents
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.probe_int_mechanisms import measured
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def worker(a,out):
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());assert audit['status']=='VERIFIED' and audit['frames']==768
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    assert digest(Path(a.checkpoint))==audit['checkpoint_sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model']);del teacher,saved
    if a.lane=='protected':
        models={}
        for n,light,period in (('protected_light',True,4),('protected_dense',False,4),('period1',True,1)):
            torch.manual_seed(303);models[n]=ProtectedMultiscaleRefreshDetector(base,light,period).cuda().eval()
        reference=None
    else:
        models={str(q):TaskPotentialEvents(base.core.coarse_update,q).cuda().eval() for q in (.125,.25,.5)}
        reference=LeakyStateUpdate(base.core.coarse_update).cuda().eval()
    for m in [base]+list(models.values())+([reference] if reference is not None else []):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};base_hash=model_digest(base);states={n:None for n in models};base_state=control=None
    rows=rows[512:544];contract=dict(lane=a.lane,frames=[r['frame'] for r in rows],checkpoint_sha256=digest(Path(a.checkpoint)),audit_sha256=digest(Path(a.audit)),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),quanta=[.125,.25,.5] if a.lane=='membrane' else [],
        boundary='Protected: closed-loop whole detector, random ordinary modules; membrane: single module on parent coarse observations, no network AP, no training, no measured timing or hardware acceleration.',
        hardware_assumptions='Signed unit events or weighted amplitude events feed a persistent 2x2 stride2 no-padding transposed-convolution accumulator. Dense candidate, warp, encoding, BN/ReLU, routing and state accesses are not free. Downstream event execution is theoretical, actual projection stays dense.')
    write_json(out/'contract.json',contract);records=[];_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];cap={}
            hooks=[base.core.observation.register_forward_pre_hook(lambda m,x:cap.update(spatial=x[0])),
                base.core.coarse_encoder.register_forward_hook(lambda m,x,y:cap.update(current=y))]
            try:expected,base_state,base_info=base.step(ego,base_state,meta)
            finally:
                for h in hooks:h.remove()
            entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            if a.lane=='protected':
                for n,m in models.items():
                    calls=[];hs=[v.register_forward_hook(lambda *_:calls.append(1)) for v in (m.core.coarse_encoder,m.core.deep_encoder)]
                    try:(pred,states[n],info),cost=measured_step(m,cap['spatial'],states[n],meta)
                    finally:
                        for h in hs:h.remove()
                    assert len(calls)==(2 if info['middle_deep_encoder_executed'] else 0)
                    assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                    if n=='period1':assert all(torch.equal(v,pred[k]) for k,v in expected.items())
                    entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],full_refresh=info['full_refresh'],heavy_calls=len(calls),
                        state_nmse={k:float((states[n][k]-base_state[k]).square().mean()/(base_state[k].square().mean()+1e-6)) for k in m.state_keys})
            else:
                x=cap['current'];bounds=block_lidar_range(base.bounds,cap['spatial'].shape[-2:],x.shape[-2:],4)
                reason,dt=reset_reason(control,meta)
                if reason:old=torch.zeros_like(x);valid=torch.zeros_like(x[:,:1],dtype=torch.bool)
                else:old,valid=warp_bev(control['value'],control['pose'],meta['pose'],bounds)
                reference.delta_t=dt;continuous=reference(x,old,valid)
                control=dict(value=continuous,pose=meta['pose'],scene=meta['segment'],timestamp_us=int(meta['timestamp_us']))
                reference_projection=base.core.deblocks[1](continuous)
                conv=base.core.deblocks[1][0]
                assert isinstance(conv,torch.nn.ConvTranspose2d) and conv.kernel_size==conv.stride==(2,2) and conv.padding==(0,0) and conv.groups==1
                fanout=conv.out_channels*4;dense_projection_macs=x.numel()*fanout
                for n,m in models.items():
                    state=states[n]
                    if reason:mv=ms=torch.zeros_like(x);valid=torch.zeros_like(x[:,:1],dtype=torch.bool)
                    else:
                        mv,valid=warp_bev(state['potential'],state['pose'],meta['pose'],bounds)
                        ms,valid_sent=warp_bev(state['sent'],state['pose'],meta['pose'],bounds);assert torch.equal(valid,valid_sent)
                    (potential,sent,counts,stats),candidate_cost=measured(m,x,mv,ms,valid,dt)
                    projection,projection_cost=measured(base.core.deblocks[1],sent)
                    assert torch.equal(potential,continuous) and stats['max_representation_error']<m.quantum+1e-5
                    assert projection_cost==2*dense_projection_macs
                    states[n]=dict(potential=potential,sent=sent,pose=meta['pose'])
                    entry['arms'][n]=dict(**stats,reset=reason,quantum=m.quantum,candidate_flops=candidate_cost,
                        actual_projection_flops=projection_cost,dense_projection_macs=dense_projection_macs,
                        theoretical_unit_ac=stats['signed_unit_events']*fanout,theoretical_weighted_event_macs=stats['active_elements']*fanout,
                        downstream_persistent_bytes=2*projection.numel()*projection.element_size(),
                        minimum_potential_sent_read_write_bytes=4*potential.numel()*potential.element_size(),
                        projection_nmse=float((projection-reference_projection).square().mean()/(reference_projection.square().mean()+1e-6)))
            log.write(json.dumps(entry)+'\n');records.append(entry)
    assert model_digest(base)==base_hash and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={}
    for n,m in models.items():
        entries=[r['arms'][n] for r in records]
        if a.lane=='protected':
            metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(x['core_conv_flops'] for x in entries)/32,
                cache_bytes=entries[0]['cache_bytes'],heavy_calls=sum(x['heavy_calls'] for x in entries),model_step_ms_mean=None,ap50=None,ap70=None)
        else:
            mean=lambda key:sum(x[key] for x in entries)/32
            metrics[n]=dict(module_parameters=sum(p.numel() for p in m.parameters()),candidate_flops_mean=mean('candidate_flops'),
                actual_projection_flops_mean=mean('actual_projection_flops'),theoretical_unit_ac_mean=mean('theoretical_unit_ac'),
                theoretical_weighted_event_macs_mean=mean('theoretical_weighted_event_macs'),dense_projection_macs_mean=mean('dense_projection_macs'),
                active_element_fraction=sum(x['active_elements'] for x in entries)/sum(x['elements'] for x in entries),unit_events_per_element=sum(x['signed_unit_events'] for x in entries)/sum(x['elements'] for x in entries),
                projection_nmse_mean=mean('projection_nmse'),state_bytes=entries[0]['state_bytes'],downstream_persistent_bytes=entries[0]['downstream_persistent_bytes'],
                model_step_ms_mean=None,ap50=None,ap70=None,scope='Task-potential candidate and one coarse deblock only; fixed tau, untrained event interface; no whole-model cost reduction measured')
    write_json(out/'report.json',dict(status='COMPLETED',lane=a.lane,scans=32,metrics=metrics,weights_unchanged=True,trained=False,ap_evaluated=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--lane',choices=('protected','membrane'),required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min state engineering')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
