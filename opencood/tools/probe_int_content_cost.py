"""Real-feature one-step content mask/halo audit, not closed-loop AP policy."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_cache import PackedPyramidStage
from opencood.models.sub_modules.int_pyramid_regions import coalesced_refresh
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def select(score,valid,fraction,tile):
    cells=F.avg_pool2d(score,tile,tile,ceil_mode=True,count_include_pad=False)
    permitted=F.max_pool2d(valid.float(),tile,tile,ceil_mode=True)>0
    eligible=torch.nonzero(permitted.flatten(),as_tuple=False).flatten();k=int(len(eligible)*fraction)
    chosen=torch.zeros_like(cells,dtype=torch.bool)
    if k:chosen.flatten()[eligible[torch.argsort(cells.flatten()[eligible],descending=True)[:k]]]=True
    expanded=chosen.repeat_interleave(tile,-2).repeat_interleave(tile,-1)[...,:valid.shape[-2],:valid.shape[-1]]
    return expanded|~valid


def worker(a):
    out=Path(a.output);audit=json.loads(Path(a.audit).read_text());assert audit['status']=='VERIFIED'
    assert digest(Path(a.checkpoint))==audit['checkpoint_sha256']
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    torch.cuda.set_per_process_memory_fraction(.5)
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    rows=rows[512:544];assert [r['frame'] for r in rows]==audit['selected_frames']
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model'],strict=True)
    for p in base.parameters():p.requires_grad_(False)
    stages={n:PackedPyramidStage(getattr(base.core,n+'_encoder'),getattr(base.core,n+'_update')).eval() for n in ('coarse','deep')}
    write_json(out/'contract.json',dict(checkpoint_sha256=digest(Path(a.checkpoint)),audit_sha256=digest(Path(a.audit)),
        frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        selection='Normalized L1 aligned fine change, pooled to target grid; fixed valid-cell/block fractions plus mandatory holes, before each local heavy call',
        fractions=[0.,.05,.1,.2],tiles=[1,4],seed=303,
        boundary='One-step operator audit on frozen-parent own states, not candidate closed-loop detection; no AP/training/timing; reference work excluded from local cost'))
    _,stream=loader(rows,cfg,0,303);state=None;records=[]
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];previous=state;capture={}
            hooks=[base.core.observation.register_forward_pre_hook(lambda m,x:capture.update(spatial=x[0])),
                base.core.coarse_encoder.register_forward_pre_hook(lambda m,x:capture.update(coarse=x[0])),
                base.core.deep_encoder.register_forward_pre_hook(lambda m,x:capture.update(deep=x[0]))]
            try:_,state,info=base.step(ego,state,meta)
            finally:
                for h in hooks:h.remove()
            fine=state['value'];bounds=block_lidar_range(base.bounds,capture['spatial'].shape[-2:],fine.shape[-2:],2)
            if info['reset']:old_fine=torch.zeros_like(fine)
            else:old_fine,_=warp_bev(previous['value'],previous['pose'],meta['pose'],bounds)
            score=(fine-old_fine).abs().mean(1,keepdim=True)/(fine.abs().mean(1,keepdim=True)+old_fine.abs().mean(1,keepdim=True)+1e-6)
            entry=dict(frame=meta['frame'],reset=info['reset'],arms={})
            for n,stage in stages.items():
                target=state[n];bounds=block_lidar_range(base.bounds,capture['spatial'].shape[-2:],target.shape[-2:],4 if n=='coarse' else 8)
                if info['reset']:old=torch.zeros_like(target);valid=torch.zeros_like(target[:,:1],dtype=torch.bool)
                else:old,valid=warp_bev(previous[n],previous['pose'],meta['pose'],bounds)
                packed=stage.pack(capture[n],old,valid);reference=stage(packed)
                assert torch.allclose(reference,target,atol=1e-4,rtol=1e-4)
                pooled=F.adaptive_avg_pool2d(score,target.shape[-2:])
                for tile,fraction in ((1,0.),(1,.05),(1,.1),(1,.2),(4,.05),(4,.1),(4,.2)):
                    request=select(pooled,valid,fraction,tile);macs=[0]
                    def count(m,inputs,output):macs[0]+=output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                    handles=[m.register_forward_hook(count) for m in stage.modules() if isinstance(m,torch.nn.Conv2d)]
                    try:
                        if bool(request.any()):actual,written,stats=coalesced_refresh(stage,packed,old,request,tile=1)
                        else:actual=old;written=request;stats=dict(stage_conv_ratio=0.,fallback=False,actual_write_fraction=0.,executed_input_area_ratio=0.)
                    finally:
                        for h in handles:h.remove()
                    assert torch.equal(request,written)
                    expected=torch.where(request,reference,old)
                    assert torch.allclose(actual,expected,atol=1e-4,rtol=1e-4)
                    entry['arms']['%s_t%d_f%s'%(n,tile,fraction)]=dict(**stats,conv_flops=2*macs[0],
                        max_error=float((actual-expected).abs().max()),requested_fraction=float(request.float().mean()),
                        holes_fraction=float((~valid).float().mean()))
            records.append(entry);log.write(json.dumps(entry)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=i+1,total=32))
    ordinary=[r for r in records if not r['reset']];metrics={}
    for n in records[0]['arms']:
        entries=[r['arms'][n] for r in ordinary]
        metrics[n]=dict(nonreset_frames=len(entries),mean_write_fraction=sum(e['actual_write_fraction'] for e in entries)/len(entries),
            mean_stage_conv_ratio=sum(e['stage_conv_ratio'] for e in entries)/len(entries),mean_conv_flops=sum(e['conv_flops'] for e in entries)/len(entries),
            fallback_frames=sum(e['fallback'] for e in entries),max_error=max(e['max_error'] for e in entries),
            module_parameters=sum(p.numel() for p in stages[n.split('_')[0]].parameters()),ap50=None,ap70=None,model_step_ms_mean=None)
    write_json(out/'report.json',dict(status='COMPLETED',scans=32,metrics=metrics,trained=False,ap_evaluated=False,timing_valid=False,
        boundary='Local stage operator geometry/cost only; parent-supplied histories, no closed-loop candidate accuracy claim'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='identity'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min content geometry budget')));signal.alarm(1800)
    try:worker(a)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
