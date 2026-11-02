"""Frozen train256 P4/P5/P6/content policy controls, threshold from train512 only."""
import argparse,copy,json,signal,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_content_refresh import ContentRefreshDetector,replay_content
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def worker(a,out):
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());cv=json.loads(Path(a.calibration_verification).read_text());gate=json.loads(Path(a.verification).read_text())
    assert audit['status']==cv['status']==gate['status']=='VERIFIED' and cv['run']=='R-160' and gate['run']=='R-139';assert digest(Path(a.checkpoint))==cv['checkpoint_sha256']==gate['checkpoint_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    ref=Path(a.trained)/'final_ap_records.json';assert digest(ref)==gate['source_sha256']['final_ap_records.json'];gold=json.loads(ref.read_text())['protected_light']
    threshold=cv['chosen']['threshold'];a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved;rows=rows[512:768]
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    models=dict(period4=ContentRefreshDetector(source),period6=copy.deepcopy(source),content=ContentRefreshDetector(source,'content',threshold));models['period6'].period=6;models['period6'].max_age=.65;models['period5']=copy.deepcopy(source);models['period5'].period=5;models['period5'].max_age=.55
    for m in [base]+list(models.values()):
        m.cuda().eval()
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];trace=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),calibration_verification_sha256=digest(Path(a.calibration_verification)),trained_sha256=gate['training']['protected_light']['checkpoint_sha256'],threshold=threshold,frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),runtime=dict(torch_version=torch.__version__,device=torch.cuda.get_device_name(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),
        boundary='Frozen R139 P4-trained weights. P4/.45 vs P5/.55 vs P6/.65 vs fixed train512 count-calibrated content(max6/.65), P5 comparator chosen by metadata-only refresh count(59 vs content58), never by P5 AP; no training or validation threshold search. Train256 diagnostic, not full validation. Fine observer always dense; content decision cannot save fine encoder. Timing invalid with instrumentation.')
    write_json(out/'contract.json',contract);dataset,stream=loader(rows,cfg,2,303);ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);assert all(torch.isfinite(v).all() for v in pred.values());entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],reset=info['reset'],evidence=info.get('evidence'))
                if meta['supervised']:
                    batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch);boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            assert all(torch.equal(states['period4']['value'],states[n]['value']) for n in ('period5','period6','content'))
            assert entry['arms']['period4']['evidence']==entry['arms']['content']['evidence']
            trace.append(dict(frame=meta['frame'],timestamp_us=int(meta['timestamp_us']),segment=meta['segment'],reset_before=meta.get('reset_before',False),evidence=entry['arms']['content']['evidence']))
            f.write(json.dumps(entry)+'\n');records.append(entry)
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=i+1,total=256))
    for n in ('period4','content'):assert replay_content(trace,n,threshold)==[r['arms'][n]['refresh'] for r in records]
    assert len(records)==256 and sum(r['supervised'] for r in records)==192 and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'selector_trace.json',trace);write_json(out/'ap_records.json',ap);raw=json.loads((out/'ap_records.json').read_text());assert raw['period4']==gold
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,orders={n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in ap.items()}))
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/256,cache_bytes=records[0]['arms'][n]['cache_bytes'],refresh_frames=sum(bool(r['arms'][n]['refresh']) for r in records),model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=256,labels=192,period4_raw_ap_exact=True,fine_states_exact=True,online_offline_exact=True,weights_unchanged=True,trained=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','calibration-verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min fixed content policy quality')));signal.alarm(2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
