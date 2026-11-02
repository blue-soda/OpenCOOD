"""Bounded channel-event Conv backend proof on fixed causal readout streams."""
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
import numpy as np
from opencood.utils import eval_utils
from opencood.tools.int_artifact_paths import artifact_path


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
    iv=json.loads(Path(a.integer_verification).read_text());assert iv['status']=='VERIFIED' and iv['run']=='R-180'
    icp=Path(a.integer_source)/'contract.json';assert digest(icp)==iv['source_sha256']['contract.json'];ic=json.loads(icp.read_text());assert ic['checkpoint_sha256']==gate['checkpoint_sha256'] and ic['calibration_sha256']==ev['calibration_sha256']
    rows=rows[512:544];models=dict(control=source,integer_dense=IntegerReadoutDetector(source,calibration['quanta'],False).cuda().eval(),integer_sparse=IntegerReadoutDetector(source,calibration['quanta'],True).cuda().eval())
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),event_verification_sha256=digest(Path(a.event_verification)),integer_verification_sha256=digest(Path(a.integer_verification)),calibration_sha256=digest(Path(a.calibration)),initialization_sha256=digest(Path(a.initialization)),frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),integer_weight_bits=16,integer_state_bits=64,precision=dict(matmul_tf32=False,cudnn_tf32=False),boundary='Frozen32 own-state nonlinear stream. Dense integer and sparse integer must be bit-exact for predictions, sending references and accumulators, including refresh. Fixed nearest integer-grid inputs at all frames and folded int16 weights change original ANN; original ANN only upstream task-state and cost control. FP32 dense core, FP64 diagnostic reference Conv and actual int64 MAC recorded separately. No AP/latency/new task-memory claim; no training.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);events=info.get('readout_events');layers=events['layers'] if events else []
                fp64=sum(v['actual_fp64_conv_flops'] for v in layers);integer=sum(v['actual_integer_macs'] for v in layers);preds[n]=pred
                row['arms'][n]=dict(core_fp32_conv_flops=cost,readout_fp64_reference_flops=fp64,readout_integer_macs=integer,core_conv_flops=cost+fp64,core_arithmetic_operations=cost+fp64+2*integer,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],events=events)
                assert all(torch.isfinite(v).all() for v in pred.values())
            row['integer_predictions_bit_exact']=all(torch.equal(preds['integer_dense'][k],preds['integer_sparse'][k]) for k in preds['integer_dense'])
            row['integer_state_bit_exact']=all(torch.equal(a,b) for key in ('readout','readout_accumulators') for a,b in zip(states['integer_dense'][key],states['integer_sparse'][key]))
            row['ann_prediction_nmse']=sum(float((preds['integer_dense'][k]-preds['control'][k]).square().mean()/(preds['control'][k].square().mean()+1e-6)) for k in preds['control'])/len(preds['control'])
            f.write(json.dumps(row)+'\n');records.append(row);assert row['integer_predictions_bit_exact'] and row['integer_state_bit_exact']
            assert all(torch.equal(states[n][k],states['control'][k]) for n in models for k in base.state_keys)
            assert row['arms']['integer_dense']['events']['layers']==[dict(v,actual_fp64_conv_flops=v['dense_mac']*2,actual_integer_macs=0,dense_reference_calls=1,reference_absolute_bound=row['arms']['integer_dense']['events']['layers'][i]['reference_absolute_bound'],event_absolute_bound=0) for i,v in enumerate(row['arms']['integer_sparse']['events']['layers'])]
            write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=32))
    assert len(records)==32 and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/32,core_fp32_conv_flops_mean=sum(r['arms'][n]['core_fp32_conv_flops'] for r in records)/32,readout_fp64_reference_flops_mean=sum(r['arms'][n]['readout_fp64_reference_flops'] for r in records)/32,readout_integer_macs_mean=sum(r['arms'][n]['readout_integer_macs'] for r in records)/32,core_arithmetic_operations_mean=sum(r['arms'][n]['core_arithmetic_operations'] for r in records)/32,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=32,weights_unchanged=True,paired_task_states_exact=True,integer_predictions_bit_exact=True,integer_states_bit_exact=True,trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','event-verification','calibration','reference-verification','integer-verification','integer-source','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.full=False;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min frozen integer nonlinear readout stream')));signal.alarm(2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
