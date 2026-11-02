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

from opencood.models.sub_modules.int_sparse_readout_events import SparseReadoutEventDetector
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
    rows=rows[512:544];models=dict(dense=ReadoutEventDetector(source,calibration['quanta']).cuda().eval(),sparse=SparseReadoutEventDetector(source,calibration['quanta']).cuda().eval())
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),event_verification_sha256=digest(Path(a.event_verification)),calibration_sha256=digest(Path(a.calibration)),initialization_sha256=digest(Path(a.initialization)),frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),atol=5e-4,rtol=1e-4,boundary='Frozen32 real frames; dense-event and sparse-event own causal sending references and accumulators. Ordinary readout Conv replaced by actual weighted channel-event MAC, BN/ReLU retained. Core cost=measured dense Conv FLOPs+2*executed sparse MAC; excludes event encoder/reductions/routing/nonlinear cost. Original output tolerance, no AP/training/valid timing. No claim of unit-AC hardware or task membrane novelty.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                calls=[];handles=[layer.register_forward_hook(lambda *_:calls.append(1)) for layer in m.core.readout.layers if isinstance(layer,nn.Conv2d)]
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta)
                for handle in handles:handle.remove()
                execution=info.get('sparse_readout_execution',[]);sparse_macs=sum(v['executed_macs'] for v in execution if not v['refresh']);preds[n]=pred
                row['arms'][n]=dict(core_conv_flops=cost+2*sparse_macs,dense_hook_flops=cost,sparse_weighted_macs=sparse_macs,readout_dense_conv_calls=len(calls),cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],events=info['readout_events'],execution=execution)
                assert all(torch.isfinite(v).all() for v in pred.values())
            row['prediction_max_abs_error']=max(float((preds['dense'][k]-preds['sparse'][k]).abs().max()) for k in preds['dense'])
            row['prediction_allclose']=all(torch.allclose(preds['dense'][k],preds['sparse'][k],atol=5e-4,rtol=1e-4) for k in preds['dense'])
            row['sending_max_abs_error']=max(float((a-b).abs().max()) for a,b in zip(states['dense']['readout'],states['sparse']['readout']))
            if a.diagnose_drift:
                row['drift_layers']=[]
                convs=[layer for layer in models['sparse'].core.readout.layers if isinstance(layer,nn.Conv2d)]
                for i,(conv,qd,qs,acc) in enumerate(zip(convs,states['dense']['readout'],states['sparse']['readout'],states['sparse']['readout_accumulators'])):
                    own_reference=conv(qs);difference=(qd-qs).abs();quantum=getattr(models['sparse'].core.readout,'quantum_%d'%i)
                    row['drift_layers'].append(dict(layer=i,sending_mismatches=int((qd!=qs).sum()),input_elements=qs.numel(),max_sending_error_in_quanta=float((difference/quantum).max()),own_accumulator_max_error=float((acc-own_reference).abs().max()),own_accumulator_allclose=bool(torch.allclose(acc,own_reference,atol=5e-4,rtol=1e-4))))
            f.write(json.dumps(row)+'\n');records.append(row)
            assert all(torch.equal(states['dense'][k],states['sparse'][k]) for k in base.state_keys)
            assert row['arms']['dense']['refresh']==row['arms']['sparse']['refresh'] and row['arms']['dense']['readout_dense_conv_calls']==4
            assert row['arms']['sparse']['readout_dense_conv_calls']==(4 if row['arms']['sparse']['refresh'] else 0)
            if row['arms']['dense']['refresh']:assert all(torch.equal(preds['dense'][k],preds['sparse'][k]) for k in preds['dense'])
            if not row['arms']['sparse']['refresh']:assert row['arms']['sparse']['sparse_weighted_macs']==sum(v['weighted_event_mac'] for v in row['arms']['sparse']['events']['layers'])
            if not a.diagnose_drift:assert row['prediction_allclose'], 'Integrated nonlinear event stream exceeds original prediction tolerance'
            write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=32))
    assert len(records)==32 and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/32,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=32,weights_unchanged=True,paired_task_states_exact=True,refresh_predictions_exact=True,drift_diagnostic=a.diagnose_drift,original_prediction_tolerance_passed=all(r['prediction_allclose'] for r in records),prediction_tolerance_failures=sum(not r['prediction_allclose'] for r in records),trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','event-verification','calibration','reference-verification','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--diagnose-drift',action='store_true',help='Preserve original tolerance failures and characterize them without claiming equivalence')
    a=p.parse_args();a.full=False;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min frozen32-frame integrated sparse readout')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
