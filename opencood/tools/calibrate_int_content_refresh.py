"""Train-only refresh-count calibration and own-state causal replay, no AP."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
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


def worker(a,out):
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());assert audit['status']==gate['status']=='VERIFIED' and gate['run']=='R-139'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'];base.load_state_dict(saved['model']);del teacher,saved;rows=rows[:512]
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();path=Path(a.trained)/'protected_light_final.pth';load_targets(source,path,gate['training']['protected_light']['checkpoint_sha256'])
    probe=ContentRefreshDetector(source).cuda().eval()
    for m in (base,source,probe):
        for p in m.parameters():p.requires_grad_(False)
    initial=model_digest(source);contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),trained_sha256=digest(path),audit_sha256=digest(Path(a.audit)),frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),runtime=dict(torch_version=torch.__version__,device=torch.cuda.get_device_name(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),
        evidence='Sum abs(current fine task state - aligned previous fine) / (sum abs(current fine)+sum abs(aligned previous fine)+1e-6), includes holes as zero history; fine independent of middle/deep refresh.',
        calibration='All unique observed training scores plus zero and max+epsilon; minimize abs(refresh count - P4 count), ties prefer underbudget then highest threshold. No labels/AP used.',
        policy='P4/.45 reference versus threshold plus max6/.65; scene/gap resets forced. Score computed after fine and before expensive middle/deep.',
        boundary='Train512 count calibration, no AP or learning. Matching calibration counts does not promise validation counts/quality. Scalar ANN scheduling is not spiking feature conversion. Fine-score reductions/synchronization are nonconv overhead.')
    write_json(out/'contract.json',contract);state=base_state=None;trace=[];_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'trace.jsonl').open('x',buffering=1) as f:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);(_,state,info),cost=measured_step(probe,x,state,meta)
            r=dict(frame=meta['frame'],timestamp_us=int(meta['timestamp_us']),segment=meta['segment'],reset_before=meta.get('reset_before',False),evidence=info['evidence'],refresh=info['full_refresh'],core_conv_flops=cost,selector_elements=info['selector_elements']);trace.append(r);f.write(json.dumps(r)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='trace',scans=i+1,total=512))
    fixed=replay_content(trace,'period4');assert fixed==[r['refresh'] for r in trace];target=sum(bool(x) for x in fixed)
    thresholds=sorted(set([0.]+[r['evidence'] for r in trace]+[max(r['evidence'] for r in trace)+1e-6]));search=[]
    for threshold in thresholds:
        count=sum(bool(x) for x in replay_content(trace,'content',threshold));search.append(dict(threshold=threshold,total_refresh=count,delta=count-target))
    best=min(search,key=lambda x:(abs(x['delta']),x['delta']>0,-x['threshold']));plans=dict(period4=fixed,content=replay_content(trace,'content',best['threshold']))
    write_json(out/'calibration.json',dict(target_refresh=target,chosen=best,search=search,plans=plans))
    models=dict(period4=ContentRefreshDetector(source),content=ContentRefreshDetector(source,'content',best['threshold']));models={n:m.cuda().eval() for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'replay.jsonl').open('x',buffering=1) as f:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);entry=dict(frame=meta['frame'],arms={})
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);assert all(torch.isfinite(v).all() for v in pred.values())
                assert info['evidence']==trace[i]['evidence'] and info['full_refresh']==plans[n][i]
                entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],age=info['age'],evidence=info['evidence'])
            assert torch.equal(states['period4']['value'],states['content']['value']);records.append(entry);f.write(json.dumps(entry)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='replay',scans=i+1,total=512))
    assert model_digest(source)==initial and all(model_digest(m)==initial for m in models.values())
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/512,refresh_frames=sum(bool(r['arms'][n]['refresh']) for r in records),cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=512,chosen=best,online_schedule_exact=True,fine_state_exact=True,weights_unchanged=True,trained=False,ap_evaluated=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min content calibration')));signal.alarm(2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
