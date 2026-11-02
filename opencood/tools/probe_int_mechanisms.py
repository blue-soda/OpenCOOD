"""Independent synthetic CUDA engineering lanes, no dataset/AP/training claims."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import torch
from torch import nn
from opencood.models.sub_modules.int_two_scale import StateUpdate
from opencood.models.sub_modules.int_mechanism_probes import SingleGateStateUpdate,FastReadoutUpdate,content_request,DeltaReadoutProbe


def write(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,sort_keys=True))
    os.replace(str(temporary),str(path))


def measured(module,*args):
    macs=[0]
    def count(m,inputs,output):macs[0]+=output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
    handles=[m.register_forward_hook(count) for m in module.modules() if isinstance(m,nn.Conv2d)]
    try:result=module(*args)
    finally:
        for h in handles:h.remove()
    return result,2*macs[0]


def metric(module,flops,**kw):
    return dict(parameters=sum(p.numel() for p in module.parameters()),core_conv_flops=flops,
        model_step_ms_mean=None,ap50=None,ap70=None,
        scope='Synthetic module only, not whole detector; no AP, timing or trained-quality result',**kw)


def state_probe():
    result={}
    for channels,h,w in ((256,24,32),(512,12,16)):
        original=StateUpdate(channels).cuda().eval();gate=SingleGateStateUpdate(original,channels).cuda().eval()
        current=torch.rand(1,channels,h,w,device='cuda');old=torch.rand_like(current)
        valid=torch.ones(1,1,h,w,device='cuda',dtype=torch.bool);valid[...,:2,:]=False
        candidate,cost0=measured(original,current,torch.where(valid,old,torch.zeros_like(old)),valid)
        output,cost1=measured(gate,current,old,valid)
        expected=torch.where(valid,(candidate+old)*.5,candidate)
        assert torch.allclose(output,expected,atol=1e-6,rtol=1e-6)
        assert torch.equal(output.masked_select(~valid),candidate.masked_select(~valid))
        output.square().mean().backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in gate.parameters() if p.grad is not None)
        assert float(gate.gate[-1].weight.grad.abs().sum())>0
        result['c%d_original'%channels]=metric(original,cost0)
        result['c%d_single_gate'%channels]=metric(gate,cost1,gate_gradient_nonzero=True,invalid_history_exact=True,
            initialization='candidate copied, gate=.5 on valid history; not identity to original update')
        with torch.no_grad():
            previous=current.clone();changed=current.clone();changed[...,h//2:h//2+2,w//2:w//2+2]+=2
            masks={}
            for fraction in (0.,.05,.1,.2):
                request,score=content_request(changed,previous,valid,fraction)
                assert bool(request.masked_select(~valid).all())
                if fraction>=.05:assert bool(request[...,h//2:h//2+2,w//2:w//2+2].all())
                masks[str(fraction)]=dict(requested_fraction=float(request.float().mean()),selected_cells=int(request.sum()),
                    mandatory_holes=int((~valid).sum()),known_changed_block_selected=bool(request[...,h//2:h//2+2,w//2:w//2+2].all()))
            result['c%d_content'%channels]=dict(synthetic_masks=masks,ap50=None,ap70=None,model_step_ms_mean=None,
                scope='Selection only, no heavy encoder execution or detection-quality claim')
    return result


def readout_probe():
    models=dict(full=nn.Sequential(nn.Conv2d(384,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU()),
        narrow=nn.Sequential(nn.Conv2d(384,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU()),
        fast=FastReadoutUpdate())
    x=torch.rand(1,384,64,96,device='cuda');old=torch.rand(1,256,64,96,device='cuda')
    valid=torch.ones(1,1,64,96,dtype=torch.bool,device='cuda');valid[...,0,:]=False
    results={}
    with torch.no_grad():
        for name,m in models.items():
            m=m.cuda().eval();y,flops=measured(m,x,old,valid) if name=='fast' else measured(m,x)
            assert y.shape==old.shape and bool(torch.isfinite(y).all())
            if name=='fast':assert torch.equal(y.masked_select(valid),old.masked_select(valid))
            results[name]=metric(m,flops,grid=[64,96],additional_readout_cache_bytes=old.numel()*old.element_size() if name=='fast' else 0,
                initialization='Random engineering weights; fast zero correction requires cache-aware learning')
    results['fast']['refresh_plus_fast_parameters']=results['full']['parameters']+results['fast']['parameters']
    return results


def delta_probe(out):
    model=DeltaReadoutProbe().cuda().eval();results=[]
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for h,w in ((16,24),(64,96)):
            for density in (0.,.001,.01,.05,.2):
                x=torch.rand(1,384,h,w,device='cuda');_,state,initial=model.step(x)
                dense_macs=initial['executed_macs']
                for step in range(1,7):
                    mask=torch.rand(1,1,h,w,device='cuda')<density
                    x=x+torch.randn_like(x)*mask*.1
                    output,state,stats=model.step(x,state);reference=model.dense(x)
                    error=float((output-reference).abs().max())
                    assert torch.allclose(output,reference,atol=1e-4,rtol=1e-4),(h,w,density,step,error)
                    record=dict(grid=[h,w],requested_input_density=density,step=step,max_error=error,
                        actual_input_cells=int(mask.sum()),dense_macs=dense_macs,**stats)
                    log.write(json.dumps(record)+'\n');results.append(record)
                write(out/'status.json',dict(status='RUNNING',phase='delta',completed_frames=len(results)))
    return dict(parameters=sum(p.numel() for p in model.parameters()),records=len(results),
        max_error=max(r['max_error'] for r in results),ap50=None,ap70=None,model_step_ms_mean=None,
        mean_executed_conv_flops=2*sum(r['executed_macs'] for r in results)/len(results),
        scope='Two Conv/ReLU aligned synthetic layers, FP32 signed vector GEMMs; not SNN or detector integration. Dense pointwise/state/scatter overhead excluded from Conv FLOPs.',
        by_density={str(d):dict(mean_conv_ratio=sum(r['executed_macs']/r['dense_macs'] for r in results if r['requested_input_density']==d)/sum(r['requested_input_density']==d for r in results),
            peak_state_bytes=max(r['state_bytes'] for r in results if r['requested_input_density']==d)) for d in (0.,.001,.01,.05,.2)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('state','readout','delta'),required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min engineering budget')));signal.alarm(1800)
    write(out/'status.json',dict(status='RUNNING',phase='cuda_gate'))
    try:
        torch.set_num_threads(1);torch.manual_seed(303)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(.35)
        smoke=nn.Conv2d(16,32,3,padding=1).cuda()(torch.ones(1,16,32,32,device='cuda'))
        torch.cuda.synchronize();assert bool(torch.isfinite(smoke).all())
        write(out/'contract.json',dict(mode=a.mode,seed=303,torch_version=torch.__version__,device=torch.cuda.get_device_name(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
            trained=False,ap_evaluated=False,timing_valid=False,synthetic=True,started_unix=time.time(),
            boundary='Engineering primitives only; no checkpoint/data migration or detector AP claim'))
        metrics=state_probe() if a.mode=='state' else readout_probe() if a.mode=='readout' else delta_probe(out)
        torch.cuda.synchronize()
        write(out/'report.json',dict(status='COMPLETED',mode=a.mode,metrics=metrics,synthetic=True,trained=False,
            ap_evaluated=False,timing_valid=False,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        write(out/'status.json',dict(status='COMPLETED'))
    except Exception as exc:
        write(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
