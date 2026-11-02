"""Fixed32 real-frame engineering, independent of the AP training queue."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import torch
from torch.nn.utils.fusion import fuse_conv_bn_eval
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector,MultiscaleRefreshDetector
from opencood.models.sub_modules.int_mechanism_probes import FastReadoutUpdate,DeltaReadoutProbe,content_request
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.probe_int_mechanisms import measured
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='input_identity'))
    audit=json.loads(Path(a.audit).read_text());assert audit['status']=='VERIFIED' and audit['frames']==768
    assert digest(Path(a.checkpoint))==audit['checkpoint_sha256']
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    torch.cuda.set_per_process_memory_fraction(.5)
    lane=a.lane;a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    rows=rows[512:544];assert [r['frame'] for r in rows]==audit['selected_frames']
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    for p in base.parameters():p.requires_grad_(False)
    del teacher,saved
    models={};states={};original_calls={}
    if lane=='state':
        models={mode:TemporalPyramidDetector(base,mode).cuda().eval() for mode in ('original','gate','leaky')}
        models.update(dual_original=MultiscaleRefreshDetector(base,'original').cuda().eval(),
            dual_gate=MultiscaleRefreshDetector(base,'gate').cuda().eval(),
            period1=MultiscaleRefreshDetector(base,'original',period=1).cuda().eval())
        states={n:None for n in models}
    elif lane=='readout':
        models=dict(fast=FastReadoutUpdate().cuda().eval());states=dict(fast=None,reuse=None)
    else:
        modules=base.core.readout.layers[0].double_conv
        model=DeltaReadoutProbe().cuda().eval()
        model.first=fuse_conv_bn_eval(copy.deepcopy(modules[0]),copy.deepcopy(modules[1]))
        model.second=fuse_conv_bn_eval(copy.deepcopy(modules[3]),copy.deepcopy(modules[4]))
        models=dict(delta=model);states=dict(delta=None)
    for model in models.values():
        for p in model.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};base_hash=model_digest(base)
    contract=dict(lane=lane,checkpoint_sha256=digest(Path(a.checkpoint)),audit_sha256=digest(Path(a.audit)),
        split='train',frames=[r['frame'] for r in rows],seed=303,scans=32,trained=False,ap_evaluated=False,timing_valid=False,
        period=4,max_age=.45,adjacent_gap_reset=.25,torch_version=torch.__version__,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        boundary='Fixed real-input engineering only; no AP or trained quality. Reference teacher work separately executed and excluded from candidate Conv budgets.')
    write_json(out/'contract.json',contract)
    _,stream=loader(rows,cfg,0,303);base_state=None;records=[];last_meta=None;last_fine=None;age=0
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];captured={}
            hooks=[base.core.observation.register_forward_pre_hook(lambda m,x:captured.update(spatial=x[0])),
                   base.core.readout.register_forward_pre_hook(lambda m,x:captured.update(readout_input=x[0])),
                   base.core.readout.register_forward_hook(lambda m,x,y:captured.update(readout_output=y))]
            try:expected,base_state,base_info=base.step(ego,base_state,meta)
            finally:
                for h in hooks:h.remove()
            spatial=captured['spatial'];entry=dict(frame=meta['frame'],supervised=meta['supervised'],reset=base_info['reset'],arms={})
            if lane=='state':
                for name,model in models.items():
                    calls=[]
                    handles=[m.register_forward_hook(lambda *_:calls.append(1)) for m in (model.core.coarse_encoder,model.core.deep_encoder)]
                    try:(pred,states[name],info),cost=measured_step(model,spatial,states[name],meta)
                    finally:
                        for h in handles:h.remove()
                    if name in ('original','period1'):assert all(torch.equal(pred[k],expected[k]) for k in pred)
                    if name.startswith('dual_'):assert len(calls)==(2 if info['full_refresh'] else 0)
                    assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                    entry['arms'][name]=dict(core_conv_flops=cost,heavy_encoder_calls=len(calls),
                        prediction_max_difference=max(float((pred[k]-expected[k]).abs().max()) for k in pred),
                        cache_bytes=info.get('cache_bytes',sum(states[name][k].numel()*states[name][k].element_size() for k in base.state_keys)),
                        full_refresh=info.get('full_refresh'),reset=info['reset'])
                fine=base_state['value'];bounds=block_lidar_range(base.bounds,spatial.shape[-2:],fine.shape[-2:],2)
                if last_fine is None or base_info['reset']:old=torch.zeros_like(fine);valid=torch.zeros_like(fine[:,:1],dtype=torch.bool)
                else:old,valid=warp_bev(last_fine,last_meta['pose'],meta['pose'],bounds)
                entry['content_masks']={}
                for fraction in (.05,.1,.2):
                    request,score=content_request(fine,old,valid,fraction)
                    assert bool(request.masked_select(~valid).all())
                    entry['content_masks'][str(fraction)]=dict(write_fraction=float(request.float().mean()),holes_fraction=float((~valid).float().mean()),score_mean=float(score.mean()))
                last_fine=fine
            elif lane=='readout':
                x,target=captured['readout_input'],captured['readout_output']
                refresh=bool(base_info['reset']) or age>=3
                age=0 if refresh else age+1
                dense_cost=2*x.shape[-2]*x.shape[-1]*9*(384*256+256*256)
                bounds=block_lidar_range(base.bounds,spatial.shape[-2:],target.shape[-2:],2)
                for name in states:
                    if refresh:result=target;cost=dense_cost;valid=torch.ones_like(target[:,:1],dtype=torch.bool)
                    else:
                        old,valid=warp_bev(states[name],last_meta['pose'],meta['pose'],bounds)
                        if name=='fast':result,cost=measured(models['fast'],x,old,valid)
                        else:result,cost=old,0
                    assert bool(torch.isfinite(result).all());states[name]=result
                    entry['arms'][name]=dict(readout_conv_flops=cost,full_refresh=refresh,readout_cache_bytes=result.numel()*result.element_size(),
                        readout_mse=float((result-target).square().mean()),readout_max_error=float((result-target).abs().max()),
                        holes_fraction=float((~valid).float().mean()))
            else:
                x,target=captured['readout_input'],captured['readout_output'];model=models['delta']
                refresh=bool(base_info['reset']) or age>=3;age=0 if refresh else age+1
                if refresh:states['delta']=None
                pred,states['delta'],stats=model.step(x,states['delta']);dense=model.dense(x)
                assert torch.allclose(dense,target,atol=1e-3,rtol=1e-4),'BN folding mismatch'
                assert torch.allclose(pred,dense,atol=1e-3,rtol=1e-4),'Incremental mismatch'
                entry['arms']['delta']=dict(**stats,prediction_max_error=float((pred-dense).abs().max()),
                    folded_reference_max_error=float((dense-target).abs().max()),
                    dense_macs=x.shape[-2]*x.shape[-1]*9*(384*256+256*256))
            log.write(json.dumps(entry)+'\n');records.append(entry);last_meta=meta
            write_json(out/'status.json',dict(status='RUNNING',phase=lane,scans=i+1,total=32))
    assert len(records)==32 and model_digest(base)==base_hash and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={}
    if lane=='state':
        for n,m in models.items():
            entries=[r['arms'][n] for r in records]
            metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(e['core_conv_flops'] for e in entries)/32,
                cache_bytes=entries[0]['cache_bytes'],heavy_encoder_calls=sum(e['heavy_encoder_calls'] for e in entries),
                model_step_ms_mean=None,ap50=None,ap70=None)
    elif lane=='readout':
        for n in states:
            entries=[r['arms'][n] for r in records];extra=sum(p.numel() for p in models['fast'].parameters()) if n=='fast' else 0
            metrics[n]=dict(parameters=32756052+extra,readout_conv_flops_mean=sum(e['readout_conv_flops'] for e in entries)/32,
                additional_readout_cache_bytes=entries[0]['readout_cache_bytes'],readout_mse_mean=sum(e['readout_mse'] for e in entries)/32,
                model_step_ms_mean=None,ap50=None,ap70=None,scope='Readout only; dense parent features computed every frame; random fast corrector not trained')
    else:
        entries=[r['arms']['delta'] for r in records]
        metrics['delta']=dict(module_parameters=sum(p.numel() for p in models['delta'].parameters()),
            readout_conv_flops_mean=2*sum(e['executed_macs'] for e in entries)/32,
            actual_vs_dense_mac_ratio=sum(e['executed_macs'] for e in entries)/sum(e['dense_macs'] for e in entries),
            max_error=max(e['prediction_max_error'] for e in entries),state_bytes=entries[0]['state_bytes'],
            model_step_ms_mean=None,ap50=None,ap70=None,scope='Real input exact vector delta on fixed grid; no warp or event quantization; dense pointwise/storage/scatter overhead excluded')
    write_json(out/'report.json',dict(status='COMPLETED',lane=lane,scans=32,labels=sum(r['supervised'] for r in records),metrics=metrics,
        weights_bn_unchanged=True,trained=False,ap_evaluated=False,timing_valid=False,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--lane',choices=('state','readout','delta'),required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min real mechanism engineering budget')));signal.alarm(1800)
    try:worker(a)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
