"""Fixed full-validation fine-cache initial/final dense/light matched controls."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector,ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_factorized_readout import compact_readout
from opencood.models.sub_modules.int_fine_refresh import FineRefreshDetector
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_artifact_paths import artifact_path
from opencood.utils import eval_utils


def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    v=json.loads(Path(a.verification).read_text());pv=json.loads(Path(a.protected_verification).read_text());rv=json.loads(Path(a.reference_verification).read_text())
    assert all(x['status']=='VERIFIED' for x in (v,pv,rv)) and (v['run'],pv['run'],rv['run'])==('R-162','R-139','R-148')
    assert digest(Path(a.checkpoint))==v['checkpoint_sha256']==pv['checkpoint_sha256'] and digest(Path(a.fine_initialization))==v['fine_initialization_sha256']
    cv=json.loads(Path(a.compact_verification).read_text());assert cv['status']=='VERIFIED' and cv['run']=='R-145'
    assert cv['checkpoint_sha256']==v['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    reference=Path(a.reference)
    for name in ('report.json','contract.json','frames.jsonl','ap_records.json'):assert digest(reference/name)==rv['source_sha256'][name]
    gold=[json.loads(x) for x in (reference/'frames.jsonl').read_text().splitlines()]
    audit=json.loads(Path(a.audit).read_text());assert audit['frames']==5794 and len(audit['files'])==7532
    write_json(out/'status.json',dict(status='RUNNING',phase='data_identity'))
    for row in audit['files']:
        p=Path(artifact_path(row['path']));assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
    a.mode='gru';a.stage='full';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance);assert [r['frame'] for r in rows]==[r['frame'] for r in gold]
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.protected_trained)/'protected_light_final.pth',pv['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    fine=torch.load(a.fine_initialization,map_location='cpu');models=dict(frozen=source)
    for tag in ('initial','final'):
        for name,light in (('fine_dense',False),('fine_light',True)):
            torch.manual_seed(303);m=FineRefreshDetector(source,light).cuda().eval();m.core.fine_update.load_state_dict(fine['fine_update']);m.core.fine_light.load_state_dict(fine['fine_light'])
            if tag=='final':load_targets(m,Path(a.trained)/(name+'_final.pth'),v['training'][name]['checkpoint_sha256'])
            models[tag+'_'+name]=m
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),protected_verification_sha256=digest(Path(a.protected_verification)),reference_verification_sha256=digest(Path(a.reference_verification)),initialization_sha256=digest(Path(a.initialization)),fine_initialization_sha256=digest(Path(a.fine_initialization)),trained_sha256={n:t['checkpoint_sha256'] for n,t in v['training'].items()},compact_verification_sha256=digest(Path(a.compact_verification)),protected_trained_sha256=pv['training']['protected_light']['checkpoint_sha256'],frames=[r['frame'] for r in rows],precision=dict(matmul_tf32=False,cudnn_tf32=False),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        boundary='Five fixed arms: frozen protected+rank128 and both initial/final fine dense/light. All R162 initial/final variants retained, no selection by diagnostic AP. Full5794 validation, not independent test. Own recurrent states; full refresh functions frozen does not imply equal states after ordinary frames. Instrumented timing invalid; strictFP32.')
    write_json(out/'contract.json',contract);dataset,stream=loader(rows,cfg,2,303);ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);assert all(torch.isfinite(y).all() for y in pred.values());preds[n]=pred
                entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info.get('full_refresh','dense'),reset=info['reset'])
            assert entry['frame']==gold[i]['frame'] and entry['supervised']==gold[i]['supervised']
            for key in ('core_conv_flops','refresh','cache_bytes'):assert entry['arms']['frozen'][key]==gold[i]['arms']['protected_initial_rank128'][key]
            for n in models:
                assert entry['arms'][n]['refresh']==entry['arms']['frozen']['refresh']
                if entry['arms']['frozen']['reset']:
                    assert all(torch.equal(preds[n][k],preds['frozen'][k]) for k in preds[n])
                    assert all(torch.equal(states[n][k],states['frozen'][k]) for k in base.state_keys)
            if meta['supervised']:
                batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch)
                for n,pred in preds.items():
                    boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            log.write(json.dumps(entry)+'\n');records.append(entry)
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='full',scans=i+1,total=5794))
    assert len(records)==5794 and sum(r['supervised'] for r in records)==1738 and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'ap_records.json',ap);write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,orders={n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in ap.items()}))
    old=json.loads((reference/'ap_records.json').read_text());new=json.loads((out/'ap_records.json').read_text())
    assert new['frozen']==old['protected_initial_rank128']
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/5794,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=5794,labels=1738,all_weights_unchanged=True,scene_reset_exact=True,prior_raw_ap_exact=True,trained=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','fine-initialization','compact-verification','initialization','protected-verification','protected-trained','reference','reference-verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min five-arm fine-cache full validation')));signal.alarm(7200)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
