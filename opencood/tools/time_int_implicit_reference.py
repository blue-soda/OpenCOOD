"""Counterbalanced explicit versus implicit integer-reference model timing."""
import argparse,copy,json,signal,subprocess,sys,os,time
from pathlib import Path
from collections import defaultdict
from opencood.models.sub_modules.int_readout_events import ReadoutEventDetector
from opencood.tools.check_int_pyramid_cache import measured_step
from torch.nn import functional as F
import torch
from torch import nn
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_fine_refresh import FineRefreshDetector
from opencood.models.sub_modules.int_factorized_readout import compact_readout
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.profile_int_compact_cache import group
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest

from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadoutDetector
from opencood.models.sub_modules.int_compact_causal_integer_state import CompactIntegerReadoutDetector
import numpy as np
from opencood.utils import eval_utils
from opencood.tools.int_artifact_paths import artifact_path


from opencood.models.sub_modules.int_subthreshold_membrane import SubthresholdMembraneDetector
from opencood.models.sub_modules.int_joint_membrane_readout import JointMembraneReadoutDetector
from opencood.models.sub_modules.int_implicit_integer_reference import ImplicitReferenceIntegerReadout,ImplicitReferenceIntegerDetector

from opencood.tools.time_int_periodic_compact import exclusive


def worker(a,out):
    uuid=os.environ["CUDA_VISIBLE_DEVICES"];exclusive(uuid)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text())
    assert (audit.get('status')=='VERIFIED' if not a.full else audit['frames']==5794 and len(audit['files'])==7532);assert gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for r in audit['files']:
        path=Path(artifact_path(r['path'])) if a.full else Path(r['relocated_path']);assert digest(path)==r['sha256']
    a.mode='gru';a.stage='full' if a.full else 'calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    ev=json.loads(Path(a.event_verification).read_text());rv=json.loads(Path(a.reference_verification).read_text());assert ev['status']==rv['status']=='VERIFIED' and ev['run']=='R-172' and rv['run']==('R-148' if a.full else 'R-168')
    event_contract=Path(a.calibration).parent/'contract.json';assert digest(event_contract)==ev['source_sha256']['contract.json'];ec=json.loads(event_contract.read_text());assert ec['verification_sha256']==digest(Path(a.verification)) and ec['initialization_sha256']==digest(Path(a.initialization)) and ev['checkpoint_sha256']==gate['checkpoint_sha256']
    assert ev['calibration_sha256']==digest(Path(a.calibration));calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    iv=json.loads(Path(a.integer_verification).read_text());assert iv['status']=='VERIFIED' and iv['run']=='R-181'
    icp=Path(a.integer_source)/'contract.json';assert digest(icp)==iv['source_sha256']['contract.json'];ic=json.loads(icp.read_text());assert ic['checkpoint_sha256']==gate['checkpoint_sha256'] and ic['calibration_sha256']==ev['calibration_sha256']
    lv=json.loads(Path(a.learning_verification).read_text());assert lv['status']=='VERIFIED' and lv['run']=='R-192';lp=Path(a.learning_source);assert digest(lp/'contract.json')==lv['source_sha256']['contract.json'];lc=json.loads((lp/'contract.json').read_text());assert lc['checkpoint_sha256']==gate['checkpoint_sha256'] and lc['initialization_sha256']==cv['rank128_checkpoint_sha256'] and lc['quantum']==.25
    path=lp/'carry_coarse_final.pth';assert digest(path)==lv['training']['carry']['checkpoint_sha256'];saved=torch.load(path,map_location='cpu');assert saved['contract']==lc and saved['mode']=='carry'
    membrane=SubthresholdMembraneDetector(source,'carry',.25).cuda().eval();membrane.core.coarse_update.load_state_dict(saved['coarse'])
    jv=json.loads(Path(a.joint_verification).read_text());assert jv['status']=='VERIFIED' and jv['run']=='R-195' and jv['carry_checkpoint_sha256']==digest(path) and jv['calibration_sha256']==ev['calibration_sha256']
    jp=Path(a.joint_source)/'contract.json';assert digest(jp)==jv['source_sha256']['contract.json']
    proof=json.loads(Path(a.implicit_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-201' and proof['checkpoint_sha256']==gate['checkpoint_sha256'] and proof['calibration_sha256']==ev['calibration_sha256'] and proof['carry_checkpoint_sha256']==digest(path)
    models=dict(integer_sparse=CompactIntegerReadoutDetector(source,calibration['quanta'],True).cuda().eval(),implicit=ImplicitReferenceIntegerDetector(source,calibration['quanta'],True).cuda().eval(),joint_sparse=JointMembraneReadoutDetector(membrane,calibration['quanta'],True).cuda().eval())
    joint=JointMembraneReadoutDetector(membrane,calibration['quanta'],True);joint.core.readout=ImplicitReferenceIntegerReadout(membrane.core.readout,calibration['quanta'],True);models['joint_implicit']=joint.cuda().eval();rows=rows[512:768];assert len(rows)==256
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()}
    contract=dict(implicit_verification_sha256=digest(Path(a.implicit_verification)),checkpoint_sha256=digest(Path(a.checkpoint)),learning_verification_sha256=digest(Path(a.learning_verification)),carry_checkpoint_sha256=digest(path),joint_verification_sha256=digest(Path(a.joint_verification)),calibration_sha256=digest(Path(a.calibration)),initialization_sha256=digest(Path(a.initialization)),
        frames=[r['frame'] for r in rows],passes=2,warmup_scans_per_pass=16,seed=303,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),uuid=uuid,device=torch.cuda.get_device_name(),torch_version=torch.__version__,precision=dict(matmul_tf32=False,cudnn_tf32=False),pass_mean_ratio_limit=1.15,
        timing='CUDA synchronize before/after model.step; includes Pillar VFE/scatter, warp, state, readout, heads and built-in event/range/membrane diagnostics. Excludes loader/H2D/AP/external equality checks; no Conv hooks. Native int64 event kernel, FP64 refresh reference; not neural hardware latency.',
        boundary='Fixed train[512:768]256 scans, two cyclic/reversed passes,16 warmups each,480 timed samples per arm. Same card exclusive guard every frame, host shared; pass mean max/min above1.15 marks timing invalid without retry. No optimizer or AP; quality separately reported. State semantics and actual event costs preserved; no statistics stripped for performance.')
    write_json(out/'contract.json',contract);records=[]
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for repeat in range(2):
            states={n:None for n in models};_,stream=loader(rows,cfg,2,303)
            for i,sample in enumerate(stream):
                exclusive(uuid);meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));names=list(models);offset=i%len(names);order=names[offset:]+names[:offset]
                if repeat:order=list(reversed(order))
                entry=dict(repeat=repeat,index=i,frame=meta['frame'],included=i>=16,execution_order=order,arms={});preds={}
                for n in order:
                    torch.cuda.synchronize();start=time.perf_counter()
                    pred,states[n],info=models[n].step(ego,states[n],meta)
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    preds[n]=pred;assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                    logical=231067468800 if info['full_refresh'] else 105679929600
                    events=info.get('readout_events');layers=events['layers'] if events else [];fp32=logical-(40461926400 if layers else 0);fp64=sum(z['actual_fp64_conv_flops'] for z in layers);mac=sum(z['actual_integer_macs'] for z in layers)
                    entry['arms'][n]=dict(model_step_ms=elapsed,full_refresh=info['full_refresh'],reset=info['reset'],logical_core_conv_flops=logical,core_fp32_conv_flops=fp32,readout_fp64_reference_flops=fp64,readout_integer_macs=mac,core_conv_flops=fp32+fp64,cache_bytes=info['cache_bytes'],events=events,membrane=info.get('membrane'))
                for x,y in (('integer_sparse','implicit'),('joint_sparse','joint_implicit')):
                    assert all(torch.equal(states[x][key],states[y][key]) for key in base.state_keys)
                    assert all(torch.equal(preds[x][key],preds[y][key]) for key in preds[x])
                    assert all(torch.equal(a,b.to(a.dtype)) for a,b in zip(states[x]['readout_accumulators'],states[y]['readout_accumulators']))
                    assert torch.equal(states[x]['readout'][0],states[y]['readout'][0]) and entry['arms'][x]['events']['layers']==entry['arms'][y]['events']['layers']
                records.append(entry);log.write(json.dumps(entry)+'\n')
                if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='timing',repeat=repeat,scans=i+1,total=256))
    exclusive(uuid);assert all(model_digest(m)==hashes[n] for n,m in models.items());metrics={}
    for n,m in models.items():
        selected=[r for r in records if r['included']];values=[r['arms'][n]['model_step_ms'] for r in selected];assert len(values)==480
        per_pass=[float(np.mean([r['arms'][n]['model_step_ms'] for r in selected if r['repeat']==rep])) for rep in range(2)]
        phases={}
        for phase,refresh in (('refresh',True),('ordinary',False)):
            group=[r['arms'][n] for r in selected if bool(r['arms'][n]['full_refresh'])==refresh];phases[phase]=dict(samples=len(group),model_step_ms_mean=float(np.mean([x['model_step_ms'] for x in group])))
        metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),samples=480,model_step_ms_mean=float(np.mean(values)),model_step_ms_p50=float(np.median(values)),model_step_ms_p95=float(np.percentile(values,95)),per_pass_ms=per_pass,pass_mean_ratio=max(per_pass)/min(per_pass),phase_metrics=phases,
            **{key+'_mean':sum(r['arms'][n][key] for r in selected)/480 for key in ('logical_core_conv_flops','core_fp32_conv_flops','readout_fp64_reference_flops','readout_integer_macs','core_conv_flops','cache_bytes')},cache_bytes_min=min(r['arms'][n]['cache_bytes'] for r in selected),cache_bytes_max=max(r['arms'][n]['cache_bytes'] for r in selected),ap50=None,ap70=None)
    valid=all(m['pass_mean_ratio']<=contract['pass_mean_ratio_limit'] for m in metrics.values())
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,timing_valid=valid,weights_unchanged=True,paired_task_states_exact=True,paired_predictions_exact=True,trained=False,ap_evaluated=False,relative_to='integer_sparse',relative_changes={n:m['model_step_ms_mean']/metrics['integer_sparse']['model_step_ms_mean']-1 for n,m in metrics.items()},boundary=contract['boundary']))
    write_json(out/'status.json',dict(status='COMPLETED',timing_valid=valid))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','event-verification','calibration','reference-verification','integer-verification','integer-source','learning-verification','learning-source','joint-verification','joint-source','implicit-verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.full=False;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min fixed paired implicit-reference timing')));signal.alarm(2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False));raise
if __name__=='__main__':main()
