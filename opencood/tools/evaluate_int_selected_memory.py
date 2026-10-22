"""Frozen e11 gate-response reuse: real recurrent AP and execution accounting."""
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
from opencood.models.sub_modules.int_selected_memory import SelectedProjectedMemory
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
    base,cfg,rows,source=setup(a);readout_best,readout=completed_best(a.readout_run)
    memory_root=Path(a.memory_run);repair_root=Path(a.repair_run)
    assert json.loads((memory_root/'status.json').read_text())['status']=='COMPLETED'
    assert json.loads((repair_root/'status.json').read_text())['status']=='COMPLETED'
    memory_best=json.loads((memory_root/'best.json').read_text());repair_best=json.loads((repair_root/'best.json').read_text())
    assert digest(memory_best['checkpoint'])==memory_best['sha256']=='1a267575ef735f203526fa3b8196334d32d9ff8a9080449c3b6206247913e5ef'
    assert digest(repair_best['checkpoint'])==repair_best['sha256']=='9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd'
    saved=torch.load(memory_best['checkpoint'],map_location='cpu');r=torch.load(repair_best['checkpoint'],map_location='cpu')
    assert saved['contract']['data']==r['contract']['data']==source['contract']
    assert saved['contract']['source']['sha256']==readout['contract']['source_sha256']==source['checkpoint']['sha256']
    assert r['contract']['readout']==readout_best
    base.shrink_conv=compact_shrink(base.shrink_conv,96)[0].cuda();base.shrink_conv.load_state_dict(readout['readout'],strict=True)
    base.feature_memory=projected_gru(base.feature_memory,32)[0].cuda()
    base.feature_memory.load_state_dict(saved['memory'],strict=True);base.eval()
    split=SplitCompactReadout(base.shrink_conv,base.backbone.deblocks[0][0].out_channels).cuda().eval()
    repair=LightRepair(96,width=32).cuda().eval();repair.load_state_dict(r['repair'],strict=True)
    roots=dict(baseline=base)
    for name in ('selected','masked_dense'):
        model=copy.deepcopy(base);model.feature_memory=SelectedProjectedMemory(base.feature_memory,.5,3,name)
        roots[name]=model.eval()
    for model in list(roots.values())+[split,repair]:
        for parameter in model.parameters():parameter.requires_grad_(False)
    modes={k:SplitReadoutDetector(m,cfg,split,3,repair) for k,m in roots.items()}
    write_json(out/'provenance.json',dict(source=source,memory=memory_best,repair=repair_best,readout=readout_best,
        threshold=.5,tile=4,dilation=1,gate_period=3,deep_period=3,
        protocol='Frozen weights. Own recurrent hidden and approximate pre-BN gate responses; retained projection references follow retained responses. Approximate reset enters candidate. Every3 frames gate responses refresh from own current hidden. No teacher cache injection. Unchanged dense E/readout/R. Selected linear MACs are explicitly added to convolution hooks.'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    arguments=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(arguments['manifest']).read_text())['train']
    costs={};states={k:None for k in modes};maximum=0.;mask_disagreements=0
    def key(name,info):return (name,bool(info['local_partial']),bool(info.get('gate_refresh',True)))
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));predictions={};details={};row_costs={}
            for name,mode in modes.items():
                counter=ConvCounter(roots[name],torch.nn.ModuleList([split,repair]))
                previous=states[name]
                try:pred,states[name],info=mode.step(ego,states[name],sample['meta'])
                finally:counter.close()
                k=key(name,info)
                if k in costs:assert costs[k]==counter.total
                costs[k]=counter.total
                row_costs[name]=counter.total
                if info['local_partial']:assert states[name]['cache']['value'] is previous['cache']['value']
                predictions[name]=pred;details[name]=info
            height,width=states['baseline']['fm']['value'].shape[-2:]
            gate_spatial_macs=sum(height*width*getattr(base.feature_memory.fusion,n)[1][1].weight.numel()
                for n in ('reset_gate_conv','update_gate_conv','candidate_conv'))
            assert row_costs['masked_dense']==row_costs['baseline']
            assert row_costs['selected']==row_costs['baseline']-(0 if details['selected']['gate_refresh'] else gate_spatial_macs)
            mask_disagreements+=int(details['selected']['gate_selected_fraction']!=details['masked_dense']['gate_selected_fraction'])
            for k in predictions['selected']:
                a1,b1=predictions['selected'][k],predictions['masked_dense'][k]
                maximum=max(maximum,float((a1-b1).abs().max()))
                assert torch.allclose(a1,b1,atol=1e-3,rtol=1e-3),maximum
    write_json(out/'engineering.json',dict(status='PASSED',frames=32,recursive_prediction_max_abs_error=maximum,
        selected_fraction_disagreements=mask_disagreements,dense_hook_macs={str(k):v for k,v in costs.items()},
        scope='Selected and masked-dense own recurrences; tolerance1e-3. Kernel roundoff may alter threshold decisions, so full validation reports both APs.'))
    states={k:None for k in modes};stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    timing={k:[] for k in modes};compute={k:0 for k in modes};fractions={k:[] for k in modes};bytes_used={k:[] for k in modes}
    refreshes={k:0 for k in modes};dataset,stream=loader(rows,cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];names=list(modes)
            offset=index%len(names);names=names[offset:]+names[:offset]
            for name in names:
                torch.cuda.synchronize();tick=time.perf_counter();pred,states[name],info=modes[name].step(ego,states[name],meta)
                torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tick)
                if index>=10:timing[name].append(elapsed)
                assert all(torch.isfinite(v).all() for v in pred.values())
                # Deep age protection can refresh on a different frame from the
                # gate period. Account for both decisions independently, rather
                # than assuming the warm-up observed every possible combination.
                dense_macs=costs[('baseline',bool(info['local_partial']),True)]
                skipped=gate_spatial_macs if name=='selected' and not info['gate_refresh'] else 0
                compute[name]+=dense_macs-skipped+info.get('gate_extra_linear_macs',0)
                fractions[name].append(info.get('gate_selected_fraction',1.))
                bytes_used[name].append(info['state_bytes']+info['anchor_bytes'])
                refreshes[name]+=int(info.get('gate_refresh',True))
                if meta['supervised']:
                    data=dict(ego=ego);boxes,scores=dataset.post.post_process(data,dict(ego=pred));gt=dataset.post.generate_gt_bbx(data)
                    for threshold in stats[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],threshold)
            if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='validation',scans=index+1))
    assert index+1==len(rows)
    results={k:dict(ap70=eval_utils.calculate_ap(stats[k],.7)[0],ap50=eval_utils.calculate_ap(stats[k],.5)[0],
        mean_ms=float(np.mean(timing[k])),p95_ms=float(np.percentile(timing[k],95)),
        conv_gflops=2*compute[k]/len(rows)/1e9,mean_gate_write_fraction=float(np.mean(fractions[k])),
        mean_state_bytes=float(np.mean(bytes_used[k])),gate_refresh_frames=refreshes[k]) for k in modes}
    assert abs(results['baseline']['ap70']-.7555532616222492)<1e-8
    write_json(out/'report.json',dict(status='COMPLETED',scans=len(rows),labels=sum(r['supervised'] for r in rows),results=results,
        scope='Own approximate recurrence, fixed .5 threshold/period3. Model wall time includes selection/warping/gather/scatter, excludes IO/H2D/postprocess. Standard convolution plus executed selected linear MACsx2; not total operator FLOPs. Masked-dense computes all spatial gates despite partial writes.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','readout-run','repair-run','memory-run','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        def expired(s,f):raise TimeoutError('Selected-memory evaluation exceeded4 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(14400)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.evaluate_int_selected_memory']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=14400,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
