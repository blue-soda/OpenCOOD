"""Training-prefix locality probe; dense recurrence, no deployed sparse claims.

Masks inspect cheap projected gate inputs. Output errors and hypothetical crop
costs are conditional on the fully computed current GRU; no errors feed back.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from opencood.models.sub_modules.int_projected_gru import projected_gru
from opencood.models.sub_modules.int_selected_conv import selected_conv3x3
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cache_repair import tile_mask,expand_tiles,rectangles,crop_jobs
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def measure_selected(layer, current, old, mask, reverse=False):
    """Isolated paired kernels; selector, warp and detector are not timed."""
    result={};timing={};macs=0
    order=('selected','dense') if reverse else ('dense','selected')
    for mode in order:
        torch.cuda.synchronize();start=time.perf_counter()
        if mode=='dense':
            value=layer(current)
        else:
            value,macs=selected_conv3x3(layer[1],current,old,mask)
        torch.cuda.synchronize();timing[mode]=1000*(time.perf_counter()-start)
        result[mode]=value
    expected=torch.where(mask,result['dense'],old)
    error=float((result['selected']-expected).abs().max())
    assert torch.isfinite(result['selected']).all()
    assert torch.allclose(result['selected'],expected,atol=1e-4,rtol=1e-4),error
    return dict(macs=macs,max_abs_error=error,dense_ms=timing['dense'],selected_ms=timing['selected'])


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    model,cfg,_,source=setup(a)
    assert digest(a.checkpoint)==a.sha256
    saved=torch.load(a.checkpoint,map_location='cpu');contract=saved['contract']
    assert contract['data']==source['contract'] and contract['source']['sha256']==source['checkpoint']['sha256']
    model.feature_memory=projected_gru(model.feature_memory,contract['rank'])[0].cuda().eval()
    model.feature_memory.load_state_dict(saved['memory'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    arguments=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    rows=json.loads(Path(arguments['manifest']).read_text())['train'][:a.frames]
    thresholds=(0.,.1,.25,.5);names=('reset_gate_conv','update_gate_conv','candidate_conv')
    write_json(out/'provenance.json',dict(source=source,checkpoint=a.checkpoint,sha256=a.sha256,epoch=saved['epoch'],
        rank=contract['rank'],thresholds=list(thresholds),tile=4,frames=len(rows),split='train_prefix',
        selected_execution=a.execute_selected,selected_threshold=.5 if a.execute_selected else None,
        protocol='Fixed checkpoint. All GRU cells run densely; each gate uses dense upstream gates. Conditional one-step errors cannot establish recursive sparse accuracy. Prior-frame projections/linear spatial outputs are nearest ego-warped. Rectangular crops are only planned. Optional selected-output kernels execute additionally at threshold .5, with dense control, numerical equality and isolated kernel timing; never substitute into recurrent state.'))
    capture={};hooks=[]
    def save(key):
        def hook(m,x,y):capture[key]=y.detach()
        return hook
    for name in names:
        gate=getattr(model.feature_memory.fusion,name)
        hooks += [gate[0].register_forward_hook(save(name+'_x')),gate[1].register_forward_hook(save(name+'_y'))]
    records={str(t):{n:[] for n in names} for t in thresholds};dense_cost=None;state=previous=None;resets=0
    selected_records={n:[] for n in names}
    bounds=cfg['model']['args']['lidar_range'];_,stream=loader(rows,cfg,2,303)
    try:
        with torch.no_grad():
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];p=ego['processed_lidar']
                if p['voxel_features'].shape[0]:
                    batch={k:p[k] for k in ('voxel_features','voxel_coords','voxel_num_points')};batch['batch_size']=1
                    spatial=model.scatter(model.pillar_vfe(batch))['spatial_features']
                else:
                    nx,ny,_=model.scatter.model_cfg['grid_size'];spatial=model.cls_head.weight.new_zeros(1,64,int(ny),int(nx))
                counter=ConvCounter(model.feature_memory,torch.nn.ModuleList()) if dense_cost is None else None
                try:_,state,info=model.feature_memory.step(spatial,state,meta)
                finally:
                    if counter:counter.close()
                if counter:dense_cost=counter.total
                reset=previous is None or info['reset'] is not None;resets+=int(reset)
                for name in names:
                    x,y=capture[name+'_x'],capture[name+'_y'];h,w=x.shape[-2:]
                    assert torch.isfinite(x).all() and torch.isfinite(y).all()
                    if reset:old_y=torch.zeros_like(y)
                    if not reset:
                        old_x,valid=warp_bev(previous[name+'_x'],previous['pose'],meta['pose'],bounds)
                        old_y,_=warp_bev(previous[name+'_y'],previous['pose'],meta['pose'],bounds)
                        score=((x-old_x).square().mean(1,keepdim=True)/torch.maximum(
                            x.square().mean(1,keepdim=True),old_x.square().mean(1,keepdim=True)).clamp_min(1e-6)).sqrt()
                        residual=(y-old_y).square().mean(1,keepdim=True);energy=y.square().mean().clamp_min(1e-12)
                    for threshold in thresholds:
                        if reset:
                            requested=written=area=1.;error=0.;jobs_count=1
                            mask=torch.ones_like(y[:,:1],dtype=torch.bool)
                        else:
                            changed=(score>threshold)|~valid
                            request=F.max_pool2d(changed.float(),3,1,1)>0
                            active=tile_mask(request,4);mask=expand_tiles(active,4,(h,w))
                            jobs,total=crop_jobs(rectangles(active),h,w,h,w,1,1,4)
                            requested=float(request.float().mean());written=float(mask.float().mean());area=total/float(h*w)
                            error=float(((residual*(~mask)).mean()/energy).sqrt());jobs_count=len(jobs)
                        records[str(threshold)][name].append([requested,written,area,error,jobs_count])
                        if a.execute_selected and threshold==.5:
                            layer=getattr(model.feature_memory.fusion,name)[1]
                            measured=measure_selected(layer,x,old_y,mask,reverse=bool(index%2))
                            expected_macs=int(mask.sum())*layer[1].in_channels*layer[1].out_channels*9
                            assert measured['macs']==expected_macs
                            selected_records[name].append(measured)
                previous=dict(capture,pose=meta['pose'])
                if (index+1)%100==0:write_json(out/'status.json',dict(status='RUNNING',phase='probe',scans=index+1))
    finally:
        for hook in hooks:hook.remove()
    assert index+1==len(rows)
    summary={}
    for threshold,by_gate in records.items():
        gates={};saving_macs=0.
        for name,values in by_gate.items():
            array=np.asarray(values);conv=getattr(model.feature_memory.fusion,name)[1][1]
            full_macs=h*w*conv.in_channels*conv.out_channels*9
            saving_macs+=full_macs*(1-array[:,2].mean())
            gates[name]=dict(mean_request=float(array[:,0].mean()),mean_write=float(array[:,1].mean()),
                mean_planned_input_area=float(array[:,2].mean()),p95_planned_input_area=float(np.percentile(array[:,2],95)),
                mean_conditional_output_relative_rms=float(array[:,3].mean()),p95_conditional_output_relative_rms=float(np.percentile(array[:,3],95)),
                mean_crops=float(array[:,4].mean()))
        summary[threshold]=dict(gates=gates,estimated_fm_conv_gflops=2*(dense_cost-saving_macs)/1e9)
    executed=None
    if a.execute_selected:
        gates={};selected_saving=0.
        for name,values in selected_records.items():
            assert len(values)==len(rows)
            conv=getattr(model.feature_memory.fusion,name)[1][1]
            full_macs=h*w*conv.in_channels*conv.out_channels*9
            mean_macs=float(np.mean([v['macs'] for v in values]));selected_saving+=full_macs-mean_macs
            warm=values[10:]
            gates[name]=dict(executed_spatial_gflops=2*mean_macs/1e9,
                executed_output_fraction=mean_macs/full_macs,
                max_masked_dense_abs_error=max(v['max_abs_error'] for v in values),
                dense_kernel_mean_ms=float(np.mean([v['dense_ms'] for v in warm])),
                selected_kernel_mean_ms=float(np.mean([v['selected_ms'] for v in warm])),
                dense_kernel_p95_ms=float(np.percentile([v['dense_ms'] for v in warm],95)),
                selected_kernel_p95_ms=float(np.percentile([v['selected_ms'] for v in warm],95)))
        executed=dict(threshold=.5,gates=gates,conditional_fm_conv_gflops=2*(dense_cost-selected_saving)/1e9,
            scope='Actual selected linear MACs plus unchanged dense projection/post-fusion MACs. Diagnostic additionally computes dense reference gates; this is NOT whole-run savings. Kernel times include gathering/chunking/scatter but exclude selector/warp/other modules. No approximate state feedback, no end-to-end latency or AP claim.')
    write_json(out/'report.json',dict(status='COMPLETED',scans=len(rows),resets=resets,
        actual_dense_fm_conv_gflops=2*dense_cost/1e9,results=summary,
        selected_execution=executed,
        scope='Estimated crop FM-only convolution includes dense projections/post_fusion plus planned spatial crops and duplicated halos. Optional separately reported selected execution is conditional, not deployed. No full-model FLOPs savings or AP claim. Output errors are pre-BN responses under dense recurrence; they do not include nonlinear/feedback accumulation.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','checkpoint','sha256','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--frames',type=int,default=1024);p.add_argument('--execute-selected',action='store_true')
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if not 32<=a.frames<=1024:p.error('--frames must be between32 and1024')
    if a.worker:
        def expired(s,f):raise TimeoutError('Locality probe exceeded two hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(7200)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.profile_int_projected_locality']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=7200,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
