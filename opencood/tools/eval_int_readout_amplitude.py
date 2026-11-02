"""Fixed train-prefix AP diagnostic after verified frozen amplitude gate."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.tools.check_int_readout_amplitude import make_models,frozen_frame
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    gate=Path(a.gate);gv=json.loads(Path(a.gate_verification).read_text());gc=json.loads((gate/'contract.json').read_text())
    assert gv['status']=='VERIFIED'
    for n in ('report.json','contract.json','frames.jsonl'):assert digest(gate/n)==gv['source_sha256'][n]
    gr=json.loads((gate/'report.json').read_text());assert gr['status']=='COMPLETED' and gr['residual_and_amplitude_bounds_checked']
    gate_rows=[json.loads(x) for x in (gate/'frames.jsonl').read_text().splitlines()]
    reference=Path(a.reference);rv=json.loads(Path(a.reference_verification).read_text());assert rv['status']=='VERIFIED'
    for n in ('report.json','contract.json','initial_frames.jsonl','initial_ap_records.json'):assert digest(reference/n)==rv['source_sha256'][n]
    rc=json.loads((reference/'contract.json').read_text());gold=[json.loads(x) for x in (reference/'initial_frames.jsonl').read_text().splitlines()]
    assert rc['checkpoint_sha256']==gc['checkpoint_sha256']==digest(Path(a.checkpoint)) and gc['scale']==rc['scale']
    data=json.loads(Path(a.data_audit).read_text());assert data['status']=='VERIFIED' and data['frames']==768
    for row in data['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    torch.cuda.set_per_process_memory_fraction(.50)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    selected=rows[512:768];assert [r['frame'] for r in selected]==rc['evaluation_frames']
    assert sorted({r[k] for r in rows[:768] for k in ('pcd','label') if r[k]})==sorted(r['relocated_path'] for r in data['files'])
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True);models={n:m.cuda().eval() for n,m in make_models(base,gc['scale']).items()}
    hashes={n:model_digest(m) for n,m in models.items()}
    assert all(sum(p.numel() for p in m.parameters())==32756052 and not any(p.requires_grad for p in m.parameters()) for m in models.values())
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),gate_report_sha256=digest(gate/'report.json'),gate_verification_sha256=digest(Path(a.gate_verification)),
        reference_report_sha256=digest(reference/'report.json'),reference_verification_sha256=digest(Path(a.reference_verification)),data_audit_sha256=digest(Path(a.data_audit)),
        evaluation_frames=[r['frame'] for r in selected],scale=gc['scale'],quantum_rule='q=s/8; no tuning or training',split='train',seed=303,
        parameters=32756052,trainable_parameters=0,scans=256,labels=192,trained=False,timing_valid=False,
        scope='Frozen conversion AP diagnostic; train subset, not full validation or independent test. Actual dense core Conv plus conditional unary/bitplane counts.')
    write_json(out/'contract.json',contract)
    states={n:None for n in models};base_state=None;records=[];ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    dataset,stream=loader(selected,cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            record,predictions=frozen_frame(models,states,x,meta)
            assert record['frame']==gold[i]['frame'] and record['supervised']==gold[i]['supervised'] and record['arms']['strict']==gold[i]['arms']['strict']
            if i<32:assert record==gate_rows[i]
            if meta['supervised']:
                data_batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data_batch)
                for name,pred in predictions.items():
                    boxes,scores=dataset.post.post_process(data_batch,dict(ego=pred))
                    for threshold in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],threshold)
            records.append(record);log.write(json.dumps(record)+'\n')
            if (i+1)%16==0:write_json(out/'status.json',dict(status='RUNNING',phase='evaluation',scans=i+1,total=256))
    assert len(records)==256 and sum(v['supervised'] for v in records)==192
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'ap_records.json',ap)
    raw=json.loads((out/'ap_records.json').read_text())
    assert raw['strict']==raw['ann']==raw['ema_reset']==json.loads((reference/'initial_ap_records.json').read_text())['strict']
    assert raw['count']==raw['residual_reset']
    metrics={n:dict(parameters=32756052,core_conv_flops_mean=sum(v['arms'][n]['core_conv_flops'] for v in records)/256,
        cache_bytes=records[0]['arms'][n]['cache_bytes'],scale_buffer_bytes=0 if n=='strict' else 1024,
        ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0],model_step_ms_mean=None,
        timing_invalid_reason='Shared GPU, activity/histogram statistics; no valid latency') for n in models}
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint_sha256=contract['checkpoint_sha256'],scans=256,labels=192,metrics=metrics,
        all_weights_bn_scale_unchanged=True,first32_gate_trace_exact=True,strict_raw_ap_replayed=True,residual_reset_count_raw_ap_exact=True,
        ann_and_ema_reset_raw_ap_exact=True,trained=False,timing_valid=False,torch_version=torch.__version__,
        boundary='Frozen parent, fixed q=s/8. No learned recovery, no AP-based scale search, no full-val/test claim. Unary/bitplane costs remain conditional and are not hardware energy measurements.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','gate','gate-verification','reference','reference-verification','data-audit','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min frozen amplitude AP budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<14500:raise RuntimeError('Requires14.5GB free; process cap50%, no valid shared-GPU timing')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.eval_int_readout_amplitude']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
