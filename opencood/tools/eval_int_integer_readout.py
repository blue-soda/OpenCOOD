"""Fixed ANN, prior float-event and integer-grid readout quality controls."""
import argparse,copy,json,signal,subprocess,sys
from pathlib import Path
from collections import defaultdict
from opencood.models.sub_modules.int_readout_events import ReadoutEventDetector
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadoutDetector
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
    ref=Path(a.reference)/('ap_records.json' if a.full else 'final_ap_records.json');assert digest(ref)==rv['source_sha256'][ref.name];gold=json.loads(ref.read_text())['protected_initial_rank128' if a.full else 'frozen']
    iv=json.loads(Path(a.integer_verification).read_text());assert iv['status']=='VERIFIED' and iv['run']=='R-181'
    icp=Path(a.integer_source)/'contract.json';assert digest(icp)==iv['source_sha256']['contract.json'];ic=json.loads(icp.read_text());assert ic['checkpoint_sha256']==gate['checkpoint_sha256'] and ic['calibration_sha256']==ev['calibration_sha256']
    prior_gate=json.loads(Path(a.event_reference_verification).read_text());assert prior_gate['status']=='VERIFIED' and prior_gate['run']==('R-176' if a.full else 'R-174');prior_path=Path(a.event_reference)/'ap_records.json';assert digest(prior_path)==prior_gate['source_sha256']['ap_records.json'];prior_ap=json.loads(prior_path.read_text())['events']
    rows=rows if a.full else rows[512:768];scans,labels=(5794,1738) if a.full else (256,192)
    models=dict(control=source,float_events=ReadoutEventDetector(source,calibration['quanta']).cuda().eval(),integer_dense=IntegerReadoutDetector(source,calibration['quanta'],False).cuda().eval())
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),event_verification_sha256=digest(Path(a.event_verification)),integer_verification_sha256=digest(Path(a.integer_verification)),event_reference_verification_sha256=digest(Path(a.event_reference_verification)),reference_verification_sha256=digest(Path(a.reference_verification)),calibration_sha256=digest(Path(a.calibration)),initialization_sha256=digest(Path(a.initialization)),frames=[r['frame'] for r in rows],full_validation=a.full,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),boundary='Fixed ANN and prior .125RMS float-event controls plus nearest integer-grid/int16 folded-weight reference. Input rounding and refresh quantization differ from old float-event method, no attribution to int16 alone. FP64 reference Conv is not deployed integer hardware. Actual FP32/F64 FLOPs, hypothetical event AC and storage separate. No training/timing or independent task-memory claim. Validation not independent test.')
    write_json(out/'contract.json',contract);dataset,stream=loader(rows,cfg,2,303);ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);preds[n]=pred;assert all(torch.isfinite(value).all() for value in pred.values())
                fp64=sum(v['actual_fp64_conv_flops'] for v in info['readout_events']['layers']) if n=='integer_dense' else 0
                row['arms'][n]=dict(core_conv_flops=cost+fp64,core_fp32_conv_flops=cost,readout_fp64_reference_flops=fp64,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],reset=info['reset'],events=info.get('readout_events'))
                if meta['supervised']:
                    batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch);boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            assert all(torch.equal(states['control'][k],states[n][k]) for n in models for k in base.state_keys)
            assert all(row['arms']['control']['core_conv_flops']==row['arms'][n]['core_conv_flops'] for n in models)
            if row['arms']['control']['refresh']:assert all(torch.equal(preds['control'][k],preds['float_events'][k]) for k in preds['control'])
            records.append(row);f.write(json.dumps(row)+'\n')
            if len(records)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=scans))
    assert len(records)==scans and sum(r['supervised'] for r in records)==labels and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'ap_records.json',ap);saved_ap=json.loads((out/'ap_records.json').read_text());assert saved_ap['control']==gold and saved_ap['float_events']==prior_ap
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,orders={n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in ap.items()}))
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/scans,core_fp32_conv_flops_mean=sum(r['arms'][n]['core_fp32_conv_flops'] for r in records)/scans,readout_fp64_reference_flops_mean=sum(r['arms'][n]['readout_fp64_reference_flops'] for r in records)/scans,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=scans,labels=labels,weights_unchanged=True,paired_task_states_exact=True,float_event_refresh_predictions_exact=True,prior_float_event_raw_ap_exact=True,control_raw_ap_exact=True,trained=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','event-verification','calibration','reference','reference-verification','integer-verification','integer-source','event-reference','event-reference-verification','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--full',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min fixed integer readout quality')));signal.alarm(7200)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
