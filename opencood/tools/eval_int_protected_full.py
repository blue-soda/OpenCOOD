"""Fixed complete validation of protected state candidates plus strict replay."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector,ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_artifact_paths import artifact_path
from opencood.utils import eval_utils


def worker(a,out):
    write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    verified=json.loads(Path(a.verification).read_text());assert verified['status']=='VERIFIED' and verified['run']=='R-139'
    reference=Path(a.reference);rv=json.loads(Path(a.reference_verification).read_text());assert rv['status']=='VERIFIED' and rv['run']=='R-112'
    for n in ('contract.json','report.json','full_frames.jsonl','full_ap_records.json'):assert digest(reference/n)==rv['source_sha256'][n]
    gold=[json.loads(x) for x in (reference/'full_frames.jsonl').read_text().splitlines()]
    assert json.loads((reference/'report.json').read_text())['torch_version']==torch.__version__
    assert digest(Path(a.checkpoint))==verified['checkpoint_sha256']
    audit=json.loads(Path(a.audit).read_text());assert audit['frames']==5794 and len(audit['files'])==7532
    write_json(out/'status.json',dict(status='RUNNING',phase='data_identity'))
    for row in audit['files']:
        p=Path(artifact_path(row['path']));assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
    torch.cuda.set_per_process_memory_fraction(.6);a.mode='gru';a.stage='full';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    assert [r['frame'] for r in rows]==[r['frame'] for r in gold]
    assert sorted({r[k] for r in rows for k in ('pcd','label') if r[k]})==sorted(artifact_path(r['path']) for r in audit['files'])
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model']);del teacher,saved
    models=dict(frozen=TemporalPyramidDetector(base,'original').cuda().eval())
    for n,light in (('protected_dense',False),('protected_light',True)):
        torch.manual_seed(303);m=ProtectedMultiscaleRefreshDetector(base,light).cuda().eval()
        load_targets(m,Path(a.trained)/(n+'_final.pth'),verified['training'][n]['checkpoint_sha256']);models[n]=m
    models['strict_replay']=AnchoredLatePyramidDetector(base,period=2,max_age=.25).cuda().eval()
    period1=ProtectedMultiscaleRefreshDetector(base,True,period=1).cuda().eval()
    load_targets(period1,Path(a.trained)/'protected_light_final.pth',verified['training']['protected_light']['checkpoint_sha256'])
    for m in [base,period1]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=p1_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),audit_sha256=digest(Path(a.audit)),
        reference_verification_sha256=digest(Path(a.reference_verification)),evaluation_frames=[r['frame'] for r in rows],
        source_trained_sha256={n:verified['training'][n]['checkpoint_sha256'] for n in ('protected_dense','protected_light')},
        split='val',seed=303,scans=5794,labels=1738,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        cost_scope='Executed core Conv2d/ConvTranspose2d=2MAC, excluding Pillar/warp/nonconv; instrumented AP evaluation is not valid latency.',
        boundary='Fixed three research arms and strict audit control, all results retained; validation-selected parent, not independent test. No period/threshold search.')
    write_json(out/'contract.json',contract);ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models};dataset,stream=loader(rows,cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});predictions={}
            assert entry['frame']==gold[i]['frame'] and entry['supervised']==gold[i]['supervised']
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                refresh=info.get('full_refresh',info.get('late_refresh'))
                if n=='frozen':refresh='dense'
                entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=refresh,reset=info['reset'])
                if n=='strict_replay':
                    prior=gold[i]['arms']['strict'];assert cost==prior['core_conv_flops'] and refresh==prior['refresh'] and info['cache_bytes']==prior['cache_bytes']
                predictions[n]=pred
            if i<32:
                expected,p1_state,_=period1.step_spatial(x,p1_state,meta)
                assert all(torch.equal(expected[k],predictions['frozen'][k]) for k in expected)
            if meta['supervised']:
                batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch)
                for n,pred in predictions.items():
                    boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            log.write(json.dumps(entry)+'\n');records.append(entry)
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='full',scans=i+1,total=5794))
    assert len(records)==5794 and sum(r['supervised'] for r in records)==1738
    assert all(model_digest(m)==hashes[n] for n,m in models.items());write_json(out/'ap_records.json',ap)
    assert json.loads((out/'ap_records.json').read_text())['strict_replay']==json.loads((reference/'full_ap_records.json').read_text())['strict']
    write_json(out/'sort_orders.json',dict(numpy_version=np.__version__,source_sha256=digest(out/'ap_records.json'),orders={n:{str(t):np.argsort(-np.asarray(v['score'])).tolist() for t,v in buckets.items()} for n,buckets in ap.items()}))
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/5794,
        cache_bytes=records[0]['arms'][n]['cache_bytes'],refresh_counts=dict(Counter(r['arms'][n]['refresh'] or 'ordinary' for r in records)),
        model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=5794,labels=1738,all_weights_bn_unchanged=True,
        strict_raw_ap_exact=True,trained_period1_first32_exact=True,timing_valid=False,trained=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','reference','reference-verification','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min protected full validation')));signal.alarm(7200)
        try:worker(a,out)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    assert int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip())<200
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.eval_int_protected_full']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
