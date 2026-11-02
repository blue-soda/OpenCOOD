"""Frozen readout AP/error by cache age, with same-weight full execution control."""
import argparse,json,signal,subprocess
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector
from opencood.models.sub_modules.int_readout_refresh import ReadoutRefreshDetector
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def worker(a,out):
    verified=json.loads(Path(a.verification).read_text());assert verified['status']=='VERIFIED' and verified['run']=='R-133'
    audit=json.loads(Path(a.audit).read_text());assert audit['status']=='VERIFIED' and audit['frames']==768
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    assert digest(Path(a.checkpoint))==verified['checkpoint_sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model']);del teacher,saved
    models=dict(frozen=TemporalPyramidDetector(base,'original').cuda().eval())
    for n,mode,source in (('dense','dense','dense'),('fast_p4','fast','fast_p4'),('fast_full','dense','fast_p4')):
        torch.manual_seed(303);m=ReadoutRefreshDetector(base,mode).cuda().eval()
        load_targets(m,Path(a.trained)/(source+'_final.pth'),verified['training'][source]['checkpoint_sha256']);models[n]=m
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;rows=rows[512:768]
    contract=dict(frames=[r['frame'] for r in rows],verification_sha256=digest(Path(a.verification)),checkpoint_sha256=digest(Path(a.checkpoint)),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),boundary='Fixed train256 error localization, not threshold search or new training. Age partitions shared by every control from fast_p4 schedule; small per-age AP is descriptive, not significance.')
    write_json(out/'contract.json',contract);groups=['all','0','1','2','3'];ap={g:{n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models} for g in groups}
    records=[];dataset,stream=loader(rows,cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));cap={}
            h=base.core.readout.register_forward_hook(lambda m,x,y:cap.update(parent=y))
            try:x,base_state=capture(base,ego,base_state,meta)
            finally:h.remove()
            predictions={};entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);predictions[n]=pred
                assert all(torch.equal(states[n][k],base_state[k]) for k in base.state_keys)
                err=float((states[n]['readout']-cap['parent']).square().mean()/(cap['parent'].square().mean()+1e-6)) if n!='frozen' else 0.
                entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],readout_nmse=err)
            entry['age']=states['fast_p4']['since_refresh'];entry['full_refresh']=bool(states['fast_p4']['since_refresh']==0)
            if meta['supervised']:
                batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch)
                for n,pred in predictions.items():
                    boxes,scores=dataset.post.post_process(batch,dict(ego=pred));entry['arms'][n]['detections']=0 if scores is None else len(scores)
                    for group in ('all',str(entry['age'])):
                        for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[group][n],t)
            records.append(entry);log.write(json.dumps(entry)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=i+1,total=256))
    original=json.loads((Path(a.trained)/'final_ap_records.json').read_text())
    for n in ('frozen','dense','fast_p4'):
        assert json.loads(json.dumps(ap['all'][n]))==original[n]
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'ap_records.json',ap)
    orders={g:{n:{str(t):np.argsort(-np.asarray(a['score'])).tolist() for t,a in buckets.items()} for n,buckets in arms.items()} for g,arms in ap.items()}
    write_json(out/'sort_orders.json',dict(numpy_version=np.__version__,source_sha256=digest(out/'ap_records.json'),orders=orders))
    metrics={}
    for group in groups:
        selected=[r for r in records if group=='all' or r['age']==int(group)]
        metrics[group]=dict(scans=len(selected),labels=sum(r['supervised'] for r in selected),models={})
        for n,m in models.items():
            metrics[group]['models'][n]=dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in selected)/len(selected),
                readout_nmse_mean=sum(r['arms'][n]['readout_nmse'] for r in selected)/len(selected),model_step_ms_mean=None,
                ap50=eval_utils.calculate_ap(ap[group][n],.5)[0],ap70=eval_utils.calculate_ap(ap[group][n],.7)[0],cache_bytes=selected[0]['arms'][n]['cache_bytes'])
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=256,weights_unchanged=True,prior_three_arm_raw_ap_exact=True,trained=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min readout localization')));signal.alarm(3600)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
