"""Frozen proof of reconstructing internal sending references from stored accumulators."""
import argparse,copy,json,signal,subprocess,sys
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

def worker(a,out):
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
    models=dict(integer_sparse=CompactIntegerReadoutDetector(source,calibration['quanta'],True).cuda().eval(),implicit=ImplicitReferenceIntegerDetector(source,calibration['quanta'],True).cuda().eval(),joint_sparse=JointMembraneReadoutDetector(membrane,calibration['quanta'],True).cuda().eval())
    joint=JointMembraneReadoutDetector(membrane,calibration['quanta'],True);joint.core.readout=ImplicitReferenceIntegerReadout(membrane.core.readout,calibration['quanta'],True);models['joint_implicit']=joint.cuda().eval();rows=rows if a.full else rows[512:544]
    total=5794 if a.full else 32
    if a.full:
        proof=json.loads(Path(a.implicit_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-201' and proof['checkpoint_sha256']==gate['checkpoint_sha256'] and proof['calibration_sha256']==ev['calibration_sha256']
        models={n:models[n] for n in ('integer_sparse','implicit')}
    pairs=[(old,new) for old,new in (('integer_sparse','implicit'),('joint_sparse','joint_implicit')) if old in models]
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(implicit_verification_sha256=digest(Path(a.implicit_verification)) if a.full else None,scans=total,full_original_only=a.full,checkpoint_sha256=digest(Path(a.checkpoint)),learning_verification_sha256=digest(Path(a.learning_verification)),carry_checkpoint_sha256=digest(path),joint_verification_sha256=digest(Path(a.joint_verification)),calibration_sha256=digest(Path(a.calibration)),initialization_sha256=digest(Path(a.initialization)),
        frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),precision=dict(matmul_tf32=False,cudnn_tf32=False),
        boundary=('Full5794 original only' if a.full else 'Frozen32 original and carry')+': compact sparse versus implicit internal sending references. Only n0 and all A persist; n1..n3 reconstructed through unchanged FP32 decode/bias/BN/ReLU/quantization. Pair outputs, task states, accumulators and external reference exact; restored previous internal inputs exact. Same event MAC, extra transient non-Conv reconstruction. No AP/model timing/optimization/energy claim; execution coverage explicitly recorded.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for old,new in pairs:
                if states[new] is not None:
                    restored=models[new].core.readout.reconstruct_references(states[new]['readout'],states[new]['readout_accumulators'])
                    assert all(torch.equal(a,b.to(a.dtype)) for a,b in zip(restored,states[old]['readout']))
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);preds[n]=pred;assert all(torch.isfinite(v).all() for v in pred.values())
                e=info['readout_events'];fp64=sum(v['actual_fp64_conv_flops'] for v in e['layers']);mac=sum(v['actual_integer_macs'] for v in e['layers'])
                row['arms'][n]=dict(core_fp32_conv_flops=cost,readout_fp64_reference_flops=fp64,readout_integer_macs=mac,core_conv_flops=cost+fp64,logical_core_conv_flops=cost+40461926400,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],events=e,membrane=info.get('membrane'))
            for old,new in pairs:
                assert all(torch.equal(preds[old][k],preds[new][k]) for k in preds[old])
                assert all(torch.equal(states[old][k],states[new][k]) for k in base.state_keys)
                assert all(torch.equal(a,b.to(a.dtype)) for a,b in zip(states[old]['readout_accumulators'],states[new]['readout_accumulators']))
                assert len(states[new]['readout'])==1 and torch.equal(states[old]['readout'][0],states[new]['readout'][0])
                assert row['arms'][old]['events']['layers']==row['arms'][new]['events']['layers']
            records.append(row);f.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=total))
    assert len(records)==total and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),**{k+'_mean':sum(r['arms'][n][k] for r in records)/total for k in ('core_conv_flops','core_fp32_conv_flops','readout_fp64_reference_flops','readout_integer_macs','logical_core_conv_flops','cache_bytes')},cache_bytes_min=min(r['arms'][n]['cache_bytes'] for r in records),cache_bytes_max=max(r['arms'][n]['cache_bytes'] for r in records),reconstructed_reference_elements_mean=sum(r['arms'][n]['events'].get('reconstructed_reference_elements',0) for r in records)/total,model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=total,weights_unchanged=True,paired_task_states_exact=True,integer_predictions_bit_exact=True,integer_state_values_exact=True,reconstructed_references_exact=True,trained=False,ap_evaluated=False,timing_valid=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','event-verification','calibration','reference-verification','integer-verification','integer-source','learning-verification','learning-source','joint-verification','joint-source','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--full',action='store_true');p.add_argument('--implicit-verification');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded frozen implicit reference proof')));signal.alarm(7200 if a.full else 2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
