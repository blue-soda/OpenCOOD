"""Fixed split-contribution repair: dense, regional and masked-dense controls."""
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
    teacher,cfg,rows,provenance=setup(a)
    readout_best,saved=completed_best(a.readout_run)
    source=Path(a.training_run)
    assert json.loads((source/'status.json').read_text())['status']=='COMPLETED'
    best=json.loads((source/'best.json').read_text());assert digest(best['checkpoint'])==best['sha256']
    trained=torch.load(best['checkpoint'],map_location='cpu');contract=trained['contract']
    assert contract['readout']==readout_best and contract['data']==provenance['contract']
    assert contract['teacher']['sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['source_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract']
    assert contract['channels']==96 and contract['period']==3
    student=copy.deepcopy(teacher)
    student.shrink_conv=compact_shrink(teacher.shrink_conv,96)[0].cuda()
    student.shrink_conv.load_state_dict(saved['readout'],strict=True);student.eval()
    split=SplitCompactReadout(student.shrink_conv,student.backbone.deblocks[0][0].out_channels).cuda().eval()
    repair=LightRepair(96,width=contract['width']).cuda().eval();repair.load_state_dict(trained['repair'],strict=True)
    for m in (teacher,student,split,repair):
        for p in m.parameters():p.requires_grad_(False)
    modes=dict(original=teacher,full=SplitReadoutDetector(student,cfg,split,1),
        cached=SplitReadoutDetector(student,cfg,split,3),dense=SplitReadoutDetector(student,cfg,split,3,repair))
    regional=[]
    for tag,threshold in (('05',.5),('10',1.)):
        for execution in ('sparse','masked_dense'):
            name=tag+'_'+execution;regional.append(name)
            modes[name]=SplitReadoutDetector(student,cfg,split,3,repair,threshold,execution)
    expected=dict(original=.758736104616861,full=.7562138415770423,cached=.752781663200643,dense=best['ap70'])
    write_json(out/'provenance.json',dict(source=provenance,repair=best,readout=readout_best,
        expected_ap70=expected,thresholds=[.5,1.],tile=4,radius=1,period=3,
        protocol='No training. Current shallow branch and spatial readout always dense; immutable deep-contribution anchors. Regional requests use current versus aligned anchor shallow descriptor and uncovered cells.'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    costs={};states={k:None for k in modes};partial_checks=0;maximum=0.;calls=[0]*5
    def hit(i):
        def hook(m,x,y):calls[i]+=1
        return hook
    hooks=[m.register_forward_hook(hit(i)) for i,m in enumerate((student.backbone.resnet.layer1,
        student.backbone.resnet.layer2,student.backbone.deblocks[1],student.backbone.deblocks[2],split.history))]
    def macs(name,info):
        if name in regional:
            if not info['local_partial']:return costs['full_full']
            return costs['cached_partial']+(costs['dense_partial']-costs['cached_partial'])*info['repair_stats']['executed_input_area_ratio']
        return costs[name+('_partial' if info.get('local_partial') else '_full')]
    _,stream=loader(rows[:32],cfg,0,303)
    try:
        with torch.no_grad():
            for sample in stream:
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];predictions={}
                for name,mode in modes.items():
                    model=teacher if name=='original' else student
                    extras=torch.nn.ModuleList([] if name=='original' else [split]+([repair] if name=='dense' or name in regional else []))
                    counter=ConvCounter(model,extras);before=list(calls);previous=states[name]
                    try:pred,states[name],info=mode.step(ego,states[name],meta)
                    finally:counter.close()
                    flag=bool(info.get('local_partial'))
                    if name in regional:
                        assert abs(counter.total-macs(name,info))<.1
                        if flag:
                            partial_checks+=1;assert calls==before
                            assert states[name]['cache']['value'] is previous['cache']['value']
                            assert states[name]['cache']['desc'] is previous['cache']['desc']
                        predictions[name]=pred
                    else:
                        key=name+('_partial' if flag else '_full')
                        if key in costs:assert costs[key]==counter.total
                        costs[key]=counter.total
                for tag in ('05','10'):
                    p,q=predictions[tag+'_sparse'],predictions[tag+'_masked_dense']
                    for k in p:
                        maximum=max(maximum,float((p[k]-q[k]).abs().max()))
                        assert torch.allclose(p[k],q[k],atol=1e-4,rtol=1e-4)
    finally:
        for hook in hooks:hook.remove()
    assert partial_checks>0
    write_json(out/'engineering.json',dict(status='PASSED',frames=32,regional_partial_checks=partial_checks,
        crop_masked_prediction_max_error=maximum,conv_macs=costs,regional_cost_hook_parity=True,immutable_anchor=True))
    states={k:None for k in modes};stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    times={k:[] for k in modes};compute={k:0 for k in modes};partial={k:0 for k in modes};areas={k:[] for k in regional}
    dataset,stream=loader(rows,cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];predictions={}
            names=list(modes);offset=index%len(names);names=names[offset:]+names[:offset]
            for name in names:
                torch.cuda.synchronize();tick=time.perf_counter()
                pred,states[name],info=modes[name].step(ego,states[name],meta)
                torch.cuda.synchronize();ms=1000*(time.perf_counter()-tick)
                assert all(torch.isfinite(v).all() for v in pred.values())
                if index>=10:times[name].append(ms)
                flag=bool(info.get('local_partial'));partial[name]+=flag;compute[name]+=macs(name,info)
                if name in regional:
                    predictions[name]=pred
                    if flag:areas[name].append([info['repair_stats'][k] for k in ('requested_fraction','actual_write_fraction','executed_input_area_ratio')])
                if meta['supervised']:
                    data=dict(ego=ego);boxes,scores=dataset.post.post_process(data,dict(ego=pred));gt=dataset.post.generate_gt_bbx(data)
                    for t in stats[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
            for tag in ('05','10'):
                p,q=predictions[tag+'_sparse'],predictions[tag+'_masked_dense']
                for k in p:
                    maximum=max(maximum,float((p[k]-q[k]).abs().max()))
                    assert torch.allclose(p[k],q[k],atol=1e-4,rtol=1e-4)
            if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='validation',scans=index+1))
    assert index+1==len(rows)
    assert len({partial[k] for k in ['cached','dense']+regional})==1
    results={k:dict(ap70=eval_utils.calculate_ap(stats[k],.7)[0],ap50=eval_utils.calculate_ap(stats[k],.5)[0],
        mean_ms=float(np.mean(times[k])),p95_ms=float(np.percentile(times[k],95)),
        conv_gflops=2*compute[k]/len(rows)/1e9,partial_frames=partial[k]) for k in modes}
    for k,ap in expected.items():assert abs(results[k]['ap70']-ap)<1e-8, 'Reloaded AP mismatch: '+k
    for k,data in areas.items():
        results[k]['partial_area_columns']=['request','write','executed_input']
        results[k]['partial_area_mean']=np.asarray(data).mean(0).tolist()
        results[k]['repair_execution_area_per_all_frames']=sum(r[2] for r in data)/len(rows)
    write_json(out/'report.json',dict(status='COMPLETED',scans=len(rows),labels=sum(r['supervised'] for r in rows),results=results,
        crop_masked_prediction_max_error=maximum,
        scope='Same GPU rotating model latency excluding IO/H2D/postprocess. Standard Conv2d/ConvTranspose2d MACs x2, not all model operations. Regional repair costs include halo duplication. Repair area per all frames counts zero on full refresh, whose heavy network work is separately included in total conv cost.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for name in ('root','readout-run','training-run','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if a.worker:
        def expired(s,f):raise TimeoutError('Regional evaluation exceeded 2 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(7200)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.evaluate_int_split_regions']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=7200,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
