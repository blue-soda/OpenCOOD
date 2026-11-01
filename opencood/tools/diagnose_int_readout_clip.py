"""Fixed-scale clipping attribution on the registered training prefix."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_if_activation import readout_coded_anchor
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def tail_stats(tensor,scales):
    """Full-map ANN diagnostic, never used to change the model or threshold."""
    assert bool((tensor>=0).all()) and tensor.shape[0]==1
    flat=tensor[0].flatten(1);limit=flat.new_tensor(scales).reshape(-1,1)
    removed=(flat-limit).clamp_min(0);mask=removed>0
    channel=dict(elements_per_channel=flat.shape[1],clipped_count=mask.sum(1).tolist(),
        activation_sum=flat.sum(1,dtype=torch.float64).tolist(),removed_sum=removed.sum(1,dtype=torch.float64).tolist(),
        activation_square_sum=flat.square().sum(1,dtype=torch.float64).tolist(),
        removed_square_sum=removed.square().sum(1,dtype=torch.float64).tolist(),maximum=flat.max(1)[0].tolist())
    spatial=removed.square().sum(0,dtype=torch.float64);total=float(spatial.sum());top=math.ceil(.01*spatial.numel())
    summary=dict(elements=flat.numel(),clipped_count=int(mask.sum()),spatial_cells=flat.shape[1],
        clipped_spatial_cells=int(mask.any(0).sum()),removed_sum=sum(channel['removed_sum']),activation_sum=sum(channel['activation_sum']),
        removed_square_sum=total,activation_square_sum=sum(channel['activation_square_sum']),
        top1pct_cells_removed_square_fraction=float(spatial.topk(top)[0].sum())/total if total else 0.,
        maximum_scale_ratio=float((flat/limit).max()))
    return dict(summary=summary,channels=channel)


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    source=Path(a.source);old=json.loads(source.read_text());scale_file=source.parent/'scales.json'
    assert old['status']=='COMPLETED' and old['scans']==256 and old['split']=='train'
    assert old['checkpoint_sha256']==digest(Path(a.checkpoint)) and old['scale_sha256']==digest(scale_file)
    scale_record=json.loads(scale_file.read_text());scales=scale_record['scales']
    assert scale_record['sample_sha256']==digest(source.parent/'activation_samples.pt')
    sampled=torch.load(source.parent/'activation_samples.pt',map_location='cpu')
    for key,value in sampled.items():
        assert torch.equal(torch.quantile(value,.999,dim=1).clamp_min(1e-6),torch.tensor(scales[key]))
    sampled_max={k:v.max(1)[0].tolist() for k,v in sampled.items()};del sampled
    gold=[json.loads(x) for x in (source.parent/'frames.jsonl').read_text().splitlines()]
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    write_json(out/'contract.json',dict(split='train',frames=256,checkpoint_sha256=digest(Path(a.checkpoint)),source_sha256=digest(source),
        scale_sha256=digest(scale_file),scales=scales,sampled_maxima=sampled_max,
        candidates=['ann','clip_input','clip_middle','clip'],policy='Frozen strict period2/max_age.25, identical caches, source R085 scales unchanged.',
        diagnostics='Full ANN input/middle maps: per-channel tail count, removed L1/L2, maxima and top1% spatial-cell error concentration; no foreground attribution.',
        engineering='All INT tests, every-frame state/core identity, first32 uncluttered ANN replay without tail hooks; ANN/both-clip raw AP records must equal R085.',
        trained=False,scale_search=False,timing_valid=False))
    models=dict(ann=AnchoredLatePyramidDetector(base),**{n:readout_coded_anchor(base,scales,n) for n in ('clip_input','clip_middle','clip')})
    states={k:None for k in models};base_state=None;records=[];ap={k:{.7:dict(tp=[],fp=[],score=[],gt=0)} for k in models}
    dataset,stream=loader(rows[:256],cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={},tails={});preds={}
            for name,model in models.items():
                hooks=[];previous=states[name]
                if name=='ann':
                    def collect(key,tensor):entry['tails'][key]=tail_stats(tensor,scales[key])
                    hooks=[model.core.readout.register_forward_pre_hook(lambda m,args:collect('input',args[0])),
                           model.core.readout.layers[0].double_conv[2].register_forward_hook(lambda m,args,y:collect('middle',y))]
                try:(pred,states[name],info),cost=measured_step(model,x,previous,meta)
                finally:
                    for hook in hooks:hook.remove()
                if name=='ann' and index<32:
                    ref,refstate,_=model.step_spatial(x,previous,meta)
                    assert all(torch.equal(pred[k],ref[k]) for k in pred)
                    assert torch.equal(states[name]['anchor']['value'],refstate['anchor']['value'])
                assert meta['frame']==gold[index]['frame'] and cost==gold[index]['arms']['ann']['core_flops']
                assert info['late_refresh']==gold[index]['arms']['ann']['refresh']
                assert all(torch.equal(states[name][k],states['ann'][k]) for k in ('value','coarse'))
                assert torch.equal(states[name]['anchor']['value'],states['ann']['anchor']['value'])
                preds[name]=pred;entry['arms'][name]=dict(core_flops=cost,refresh=info['late_refresh'])
            if meta['supervised']:
                data=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data)
                for name,pred in preds.items():
                    start=len(ap[name][.7]['tp']);boxes,scores=dataset.post.post_process(data,dict(ego=pred))
                    eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],.7)
                    entry['arms'][name]['detections']=dict(gt=int(gt.shape[0]),tp=sum(ap[name][.7]['tp'][start:]),fp=sum(ap[name][.7]['fp'][start:]))
            entry['all_cache_states_equal_ann']=True;records.append(entry);log.write(json.dumps(entry)+'\n')
            if (index+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='clip_diagnostic',scans=index+1))
    assert len(records)==256
    write_json(out/'ap_records.json',ap)
    old_ap=json.loads((source.parent/'ap_records.json').read_text())
    for name in ('ann','clip'):assert ap[name][.7]==old_ap[name]['0.7']
    metrics={k:dict(ap70=eval_utils.calculate_ap(v,.7)[0],tp=sum(v[.7]['tp']),fp=sum(v[.7]['fp']),gt=v[.7]['gt']) for k,v in ap.items()}
    summaries={}
    for key in scales:
        rows_=[r['tails'][key]['summary'] for r in records]
        sums={name:sum(r[name] for r in rows_) for name in ('elements','clipped_count','spatial_cells','clipped_spatial_cells','removed_sum','activation_sum','removed_square_sum','activation_square_sum')}
        summaries[key]=dict(sums,clipped_fraction=sums['clipped_count']/sums['elements'],removed_l1_fraction=sums['removed_sum']/sums['activation_sum'],
            removed_l2_fraction=sums['removed_square_sum']/sums['activation_square_sum'],
            maximum_scale_ratio=max(r['maximum_scale_ratio'] for r in rows_),
            mean_top1pct_cells_removed_square_fraction=sum(r['top1pct_cells_removed_square_fraction'] for r in rows_)/256,
            full_map_maxima=[max(r['tails'][key]['channels']['maximum'][i] for r in records) for i in range(len(scales[key]))])
    write_json(out/'report.json',dict(status='COMPLETED',split='train',scans=256,labels=sum(r['supervised'] for r in records),
        checkpoint_sha256=digest(Path(a.checkpoint)),scale_sha256=digest(scale_file),metrics=metrics,tail_summaries=summaries,
        ann_and_both_clip_raw_ap_equal_source=True,all_cache_states_and_core_equal_ann=True,no_tail_hook_prediction_exact_frames=32,
        caveat='Fixed-scale train diagnostic only; separate clipping effects are not additive. Spatial concentration is not object/foreground attribution. No training, sparse execution, timing or independent-validation conclusion.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','source'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h fixed-scale diagnostic budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('At least12GB free required')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.diagnose_int_readout_clip']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
