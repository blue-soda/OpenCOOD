"""Fixed complete validation of compact readout initial/final and cache combination."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector,ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_factorized_readout import compact_readout
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
    assert all(x['status']=='VERIFIED' for x in (v,pv,rv)) and (v['run'],pv['run'],rv['run'])==('R-147','R-139','R-140')
    assert digest(Path(a.checkpoint))==v['checkpoint_sha256']==pv['checkpoint_sha256'] and digest(Path(a.initialization))==v['initialization_sha256']
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
    models={n:TemporalPyramidDetector(base,'original').cuda().eval() for n in ('frozen','initial_rank128','final_dense','final_rank128')}
    for n in ('protected','protected_initial_rank128'):
        m=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(m,Path(a.protected_trained)/'protected_light_final.pth',pv['training']['protected_light']['checkpoint_sha256']);models[n]=m
    initial=torch.load(a.initialization,map_location='cpu')['readout']
    for n in ('initial_rank128','final_rank128','protected_initial_rank128'):
        models[n].core.readout,_=compact_readout(base.core.readout,128);models[n].core.readout.load_state_dict(initial);models[n].eval()
    for n,source in (('final_dense','dense'),('final_rank128','rank128')):
        path=Path(a.trained)/(source+'_readout_final.pth');assert digest(path)==v['training'][source]['checkpoint_sha256'];models[n].core.readout.load_state_dict(torch.load(path,map_location='cpu')['readout'])
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),protected_verification_sha256=digest(Path(a.protected_verification)),reference_verification_sha256=digest(Path(a.reference_verification)),initialization_sha256=digest(Path(a.initialization)),readout_trained_sha256={n:r['checkpoint_sha256'] for n,r in v['training'].items()},protected_trained_sha256=pv['training']['protected_light']['checkpoint_sha256'],frames=[r['frame'] for r in rows],precision=dict(matmul_tf32=False,cudnn_tf32=False),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        boundary='Fixed six arms, retains initial and final despite adaptation decline. Compact+protected is a fixed untrained composition, not a separately trained optimum. Full5794 validation, not independent test; no threshold search. Latency invalid in instrumented AP run. Prior default precision AP raw replay recorded, not assumed.')
    write_json(out/'contract.json',contract);dataset,stream=loader(rows,cfg,2,303);ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);assert all(torch.isfinite(y).all() for y in pred.values());preds[n]=pred
                entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info.get('full_refresh','dense'),reset=info['reset'])
            assert entry['frame']==gold[i]['frame'] and entry['supervised']==gold[i]['supervised']
            for k in base.state_keys:
                assert all(torch.equal(states['frozen'][k],states[n][k]) for n in ('initial_rank128','final_dense','final_rank128'))
                assert torch.equal(states['protected'][k],states['protected_initial_rank128'][k])
            for k in ('core_conv_flops','refresh','cache_bytes'):assert entry['arms']['protected'][k]==gold[i]['arms']['protected_light'][k]
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
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/5794,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=5794,labels=1738,all_weights_unchanged=True,paired_upstream_states_exact=True,prior_raw_ap_replay=dict(frozen=new['frozen']==old['frozen'],protected=new['protected']==old['protected_light']),trained=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','initialization','protected-verification','protected-trained','reference','reference-verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min six-arm compact full validation')));signal.alarm(7200)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
