"""Real motion: execute sparse projection events and account for exact correction."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason
from opencood.models.sub_modules.int_task_membrane import TaskPotentialEvents
from opencood.models.sub_modules.int_event_projection import sparse_deconv_events,warp_projection_phases
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.tools.probe_int_mechanisms import measured
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def worker(a,out):
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text())
    assert audit['status']==gate['status']=='VERIFIED' and gate['run']=='R-135' and digest(Path(a.checkpoint))==gate['checkpoint_sha256']
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model']);del teacher,saved
    cell=TaskPotentialEvents(base.core.coarse_update,.25).cuda().eval();conv=base.core.deblocks[1][0];tail=base.core.deblocks[1][1:]
    for m in (base,cell):
        for p in m.parameters():p.requires_grad_(False)
    hashes=[model_digest(base),model_digest(cell)];rows=rows[512:544];state=base_state=None;records=[]
    contract=dict(frames=[r['frame'] for r in rows],checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),
        quantum=.25,tau=.2,projection_period=4,max_age=.45,phase_warp=a.phase_warp,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        boundary='Single task-state module on parent observations, no network AP/training/timing. Exact candidate really computes dense alignment correction plus sparse signed projection events; naive no-correction path is an invalid-cost comparison.',
        equation='D(s_new)=Wout(z_prev)+[D(Win(s_prev))-Wout(z_prev)]+D(delta). Bracketed correction requires dense D for general poses; not free.',
        state='Candidate retains potential, sent input, linear projection accumulator; naive accumulator is a separate diagnostic-only buffer, not candidate state.')
    if a.phase_warp:
        contract.update(boundary='Single task-state module, no AP/training/timing. Candidate executes input-grid phase-preserving output warp plus actual sparse signed projection events. Full current projection is audit-only.',
            equation='D(s_new)=Wphase(z_prev)+D_sparse(delta); Wphase(D(s))=D(Win(s)) for this non-overlapping bias-free 2x2/stride2 projection. Dense phase warp/reshapes, BN/ReLU and memory traffic are not free.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];cap={}
            hooks=[base.core.observation.register_forward_pre_hook(lambda m,x:cap.update(spatial=x[0])),base.core.coarse_encoder.register_forward_hook(lambda m,x,y:cap.update(current=y))]
            try:_,base_state,_=base.step(ego,base_state,meta)
            finally:
                for h in hooks:h.remove()
            x=cap['current'];reason,dt=reset_reason(state,meta);stamp=int(meta['timestamp_us']);refresh=reason
            if refresh is None:
                if state['age']+1>=4:refresh='period'
                elif (stamp-state['refresh_stamp'])/1e6>=.45-1e-6:refresh='age'
            bounds=block_lidar_range(base.bounds,cap['spatial'].shape[-2:],x.shape[-2:],4)
            if reason:mv=ms=torch.zeros_like(x);valid=torch.zeros_like(x[:,:1],dtype=torch.bool)
            else:
                mv,valid=warp_bev(state['potential'],state['pose'],meta['pose'],bounds)
                ms,vs=warp_bev(state['sent'],state['pose'],meta['pose'],bounds);assert torch.equal(valid,vs)
            (v,sent,counts,events),candidate_flops=measured(cell,x,mv,ms,valid,dt)
            reference=conv(sent);dense_macs=sent.numel()*conv.out_channels*4
            correction_max=0.;event_macs=0;naive_one_error=0.;commutator_nmse=0.
            if refresh:
                z,full_flops=measured(conv,sent);naive_z=z.clone();assert full_flops==2*dense_macs
            else:
                output_bounds=block_lidar_range(base.bounds,cap['spatial'].shape[-2:],state['z'].shape[-2:],2)
                warped,_=warp_bev(state['z'],state['pose'],meta['pose'],output_bounds)
                naive_warped,_=warp_bev(state['naive_z'],state['pose'],meta['pose'],output_bounds)
                if a.phase_warp:
                    aligned_projection,pvalid=warp_projection_phases(state['z'],state['pose'],meta['pose'],bounds)
                    assert torch.equal(valid,pvalid);full_flops=0
                else:aligned_projection,full_flops=measured(conv,ms)
                correction=aligned_projection-warped
                delta,stats=sparse_deconv_events(conv,.25*counts);event_macs=stats['weighted_event_macs']
                assert stats['active_elements']==events['active_elements']
                z=aligned_projection+delta if a.phase_warp else warped+correction+delta;naive_z=naive_warped+delta
                correction_max=float(correction.abs().max());commutator_nmse=float(correction.square().mean()/(aligned_projection.square().mean()+1e-6))
                naive_one_error=float((tail(warped+delta)-tail(reference)).square().mean()/(tail(reference).square().mean()+1e-6))
            actual=tail(z);target=tail(reference);naive=tail(naive_z)
            assert torch.allclose(z,reference,atol=2e-4,rtol=1e-4) and torch.allclose(actual,target,atol=1e-3,rtol=1e-4)
            stored=sum(t.numel()*t.element_size() for t in (v,sent,z))
            entry=dict(frame=meta['frame'],supervised=meta['supervised'],refresh=refresh,candidate_flops=candidate_flops,
                dense_projection_macs=dense_macs,alignment_or_refresh_flops=full_flops,weighted_event_macs=event_macs,
                valid_projection_flops=full_flops+2*event_macs,actual_candidate_plus_projection_flops=candidate_flops+full_flops+2*event_macs,
                corrected_max_error=float((actual-target).abs().max()),naive_one_step_nmse=naive_one_error,
                naive_closed_projection_nmse=float((naive-target).square().mean()/(target.square().mean()+1e-6)),
                commutator_nmse=commutator_nmse,correction_max_abs=correction_max,state_bytes=stored,
                diagnostic_only_naive_state_bytes=naive_z.numel()*naive_z.element_size(),minimum_state_read_write_bytes=2*stored,
                active_elements=events['active_elements'],signed_unit_events=events['signed_unit_events'])
            state=dict(potential=v,sent=sent,z=z,naive_z=naive_z,pose=meta['pose'],scene=meta['segment'],timestamp_us=stamp,
                age=0 if refresh else state['age']+1,refresh_stamp=stamp if refresh else state['refresh_stamp'])
            records.append(entry);log.write(json.dumps(entry)+'\n')
    assert [model_digest(base),model_digest(cell)]==hashes
    mean=lambda k:sum(r[k] for r in records)/32;ordinary=[r for r in records if not r['refresh']]
    metrics=dict(module_parameters=sum(p.numel() for p in cell.parameters())+sum(p.numel() for p in conv.parameters()),
        actual_candidate_plus_projection_flops_mean=mean('actual_candidate_plus_projection_flops'),candidate_flops_mean=mean('candidate_flops'),
        valid_projection_flops_mean=mean('valid_projection_flops'),dense_projection_flops_mean=2*mean('dense_projection_macs'),
        weighted_event_macs_mean=mean('weighted_event_macs'),state_bytes=records[0]['state_bytes'],
        max_corrected_error=max(r['corrected_max_error'] for r in records),ordinary_commutator_nmse=sum(r['commutator_nmse'] for r in ordinary)/len(ordinary),
        ordinary_naive_one_step_nmse=sum(r['naive_one_step_nmse'] for r in ordinary)/len(ordinary),ordinary_naive_closed_nmse=sum(r['naive_closed_projection_nmse'] for r in ordinary)/len(ordinary),
        model_step_ms_mean=None,ap50=None,ap70=None)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=32,refresh_frames=32-len(ordinary),ordinary_frames=len(ordinary),weights_unchanged=True,trained=False,ap_evaluated=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--phase-warp',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min event alignment engineering')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
