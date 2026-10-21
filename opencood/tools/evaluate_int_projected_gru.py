"""Untrained SVD initialization audit for locally structured GRU baselines."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_compact_readout import compact_shrink
from opencood.models.sub_modules.int_projected_gru import projected_gru
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout,SplitReadoutDetector
from opencood.models.sub_modules.int_light_repair import LightRepair
from opencood.tools.evaluate_int_compact_repair import completed_best
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    teacher,cfg,rows,source=setup(a);best,readout=completed_best(a.readout_run)
    initial=Path(a.repair_run);assert json.loads((initial/'status.json').read_text())['status']=='COMPLETED'
    repair_best=json.loads((initial/'best.json').read_text());assert digest(repair_best['checkpoint'])==repair_best['sha256']
    assert repair_best['sha256']=='9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd'
    saved=torch.load(repair_best['checkpoint'],map_location='cpu')
    assert saved['contract']['readout']==best and saved['contract']['data']==source['contract']
    assert readout['contract']['source_sha256']==saved['contract']['teacher']['sha256']==source['checkpoint']['sha256']
    base=copy.deepcopy(teacher);base.shrink_conv=compact_shrink(teacher.shrink_conv,96)[0].cuda()
    base.shrink_conv.load_state_dict(readout['readout'],strict=True);base.eval()
    split=SplitCompactReadout(base.shrink_conv,base.backbone.deblocks[0][0].out_channels).cuda().eval()
    repair=LightRepair(96,width=32).cuda().eval();repair.load_state_dict(saved['repair'],strict=True)
    roots=dict(original=teacher,base=base);modes=dict(original=teacher,
        base_full=SplitReadoutDetector(base,cfg,split,1),base_cached=SplitReadoutDetector(base,cfg,split,3,repair))
    owner=dict(original='original',base_full='base',base_cached='base');energy={};parameters={}
    for rank in (32,64):
        tag='r%d'%rank;model=copy.deepcopy(base)
        memory,energy[tag]=projected_gru(base.feature_memory,rank);model.feature_memory=memory.cuda().eval()
        parameters[tag]=sum(p.numel() for p in memory.parameters());roots[tag]=model
        modes[tag+'_full']=SplitReadoutDetector(model,cfg,split,1)
        modes[tag+'_cached']=SplitReadoutDetector(model,cfg,split,3,repair)
        owner[tag+'_full']=owner[tag+'_cached']=tag
    for m in list(roots.values())+[split,repair]:
        for p in m.parameters():p.requires_grad_(False)
    parameters['base']=sum(p.numel() for p in base.feature_memory.parameters())
    expected=dict(original=.758736104616861,base_full=.7562138415770423,base_cached=.7553967658970537)
    write_json(out/'provenance.json',dict(source=source,readout=best,repair=repair_best,retained_weight_energy=energy,
        memory_parameters=parameters,expected_ap70=expected,period=3,max_age=.25,
        protocol='No training; SVD rank32/64 on reset/update/candidate input channels only. Same BN/nonlinearity, post-fusion and state lattice. Student owns recurrent state and full anchors. All GRU cells evaluated densely. Cached path transfers the fixed dense R, not the failed sparse-adaptation weights.'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    states={k:None for k in modes};costs={};partials=0
    _,stream=loader(rows[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            for name,mode in modes.items():
                extras=[] if name=='original' else [split]+([repair] if name.endswith('_cached') else [])
                counter=ConvCounter(roots[owner[name]],torch.nn.ModuleList(extras));previous=states[name]
                try:pred,states[name],info=mode.step(ego,states[name],meta)
                finally:counter.close()
                assert all(torch.isfinite(v).all() for v in pred.values())
                partial=bool(info.get('local_partial'));key=name+('_partial' if partial else '_full')
                if key in costs:assert costs[key]==counter.total
                costs[key]=counter.total
                if partial:
                    partials+=1;assert states[name]['cache']['value'] is previous['cache']['value']
    assert partials>0
    for tag in ('r32','r64'):
        assert costs[tag+'_full_full']<costs['base_full_full']
        assert costs[tag+'_cached_partial']<costs['base_cached_partial']
    write_json(out/'engineering.json',dict(status='PASSED',frames=32,partial_checks=partials,
        conv_macs=costs,immutable_anchors=True,finite_predictions=True))
    states={k:None for k in modes};stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    timing={k:[] for k in modes};compute={k:0 for k in modes};partial={k:0 for k in modes}
    dataset,stream=loader(rows,cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            names=list(modes);offset=index%len(names);names=names[offset:]+names[:offset]
            for name in names:
                torch.cuda.synchronize();tick=time.perf_counter();pred,states[name],info=modes[name].step(ego,states[name],meta)
                torch.cuda.synchronize();ms=1000*(time.perf_counter()-tick)
                if index>=10:timing[name].append(ms)
                assert all(torch.isfinite(v).all() for v in pred.values())
                flag=bool(info.get('local_partial'));partial[name]+=flag;compute[name]+=costs[name+('_partial' if flag else '_full')]
                if meta['supervised']:
                    data=dict(ego=ego);boxes,scores=dataset.post.post_process(data,dict(ego=pred));gt=dataset.post.generate_gt_bbx(data)
                    for t in stats[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
            if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='validation',scans=index+1))
    assert index+1==len(rows) and len({partial[k] for k in modes if k.endswith('_cached')})==1
    results={k:dict(ap70=eval_utils.calculate_ap(stats[k],.7)[0],ap50=eval_utils.calculate_ap(stats[k],.5)[0],
        mean_ms=float(np.mean(timing[k])),p95_ms=float(np.percentile(timing[k],95)),
        conv_gflops=2*compute[k]/len(rows)/1e9,partial_frames=partial[k]) for k in modes}
    for name,value in expected.items():assert abs(results[name]['ap70']-value)<1e-8
    write_json(out/'report.json',dict(status='COMPLETED',scans=len(rows),labels=sum(r['supervised'] for r in rows),results=results,
        scope='Untrained SVD initialization, not a convergence or capacity test. Same GPU rotating model latency excluding IO/H2D/postprocess. Standard convolution MACs x2 only. Dense projected GRU; no GRU region skipping.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','readout-run','repair-run','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        def expired(s,f):raise TimeoutError('Projected GRU evaluation exceeded two hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(7200)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.evaluate_int_projected_gru']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=7200,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
