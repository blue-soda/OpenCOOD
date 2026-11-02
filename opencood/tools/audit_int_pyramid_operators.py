"""Frozen strict-anchor per-module cost audit on the fixed pilot evaluation stream."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def category(name):
    parts=name.split('.')
    if parts[0]!='core':return parts[0]
    if parts[1].endswith('_update'):return '.'.join(parts[1:3])
    if parts[1]=='deblocks':return '.'.join(parts[1:3])
    return parts[1]


def profiled_step(model,x,state,meta):
    calls=[]
    def hook(name):
        def count(module,inputs,output):
            inp=inputs[0];transposed=isinstance(module,torch.nn.ConvTranspose2d)
            macs=(inp.numel()*module.out_channels//module.groups if transposed else output.numel()*module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            calls.append(dict(name=name,category=category('core.'+name),input_shape=list(inp.shape),output_shape=list(output.shape),
                core_conv_flops=2*macs,input_elements=inp.numel(),input_nonzero=int(torch.count_nonzero(inp)),
                input_min=float(inp.min()),input_max=float(inp.max())))
        return count
    hooks=[m.register_forward_hook(hook(n)) for n,m in model.core.named_modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.ConvTranspose2d))]
    try:result,cost=measured_step(model,x,state,meta)
    finally:
        for h in hooks:h.remove()
    assert sum(c['core_conv_flops'] for c in calls)==cost
    return result,cost,calls


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    source=Path(a.reference);previous=json.loads((source/'report.json').read_text());prior_contract=json.loads((source/'contract.json').read_text())
    assert previous['status']=='COMPLETED' and previous['readout_only'] and previous['matched_reference_verified']
    assert previous['checkpoint_sha256']==digest(Path(a.checkpoint))
    gold=[json.loads(line) for line in (source/'final_frames.jsonl').read_text().splitlines()]
    torch.cuda.set_per_process_memory_fraction(.45)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    rows=rows[512:768];assert [r['frame'] for r in rows]==prior_contract['evaluation_frames']
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    assert saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True);model=AnchoredLatePyramidDetector(base);before=model_digest(model)
    parameter_rows=[dict(name=n,shape=list(p.shape),elements=p.numel(),category=category(n)) for n,p in model.named_parameters()]
    assert sum(p['elements'] for p in parameter_rows)==32756052
    inventory={n:dict(type=type(m).__name__,in_channels=m.in_channels,out_channels=m.out_channels,groups=m.groups,
        kernel=list(m.kernel_size),stride=list(m.stride),padding=list(m.padding),dilation=list(m.dilation),
        weight_shape=list(m.weight.shape),bias=m.bias is not None) for n,m in model.core.named_modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.ConvTranspose2d))}
    write_json(out/'inventory.json',dict(parameters=parameter_rows,convolutions=inventory))
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),reference_report_sha256=digest(source/'report.json'),
        reference_frames_sha256=digest(source/'final_frames.jsonl'),frames=prior_contract['evaluation_frames'],split='train',seed=303,
        scope='Core Conv2d/ConvTranspose2d multiply and add=2 FLOPs; excludes Pillar/BN/ReLU/warp/cache/IO/postprocess; bias excluded. Parameter count includes entire detector.',
        input_activity='Exact nonzero values at each invoked convolution input; continuous values, NOT binary events or an AC estimate.',
        trained=False,timing_valid=False,budget_minutes=30)
    write_json(out/'contract.json',contract)
    dataset,stream=loader(rows,cfg,2,303);state=base_state=None;records=[];ap={t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)}
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            (pred,state,info),cost,calls=profiled_step(model,x,state,meta)
            assert meta['frame']==gold[i]['frame'] and meta['supervised']==gold[i]['supervised']
            assert dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['late_refresh'],memory=None)==gold[i]['arms']['strict']
            assert all(bool(torch.isfinite(v).all()) for v in pred.values())
            cache={n:dict(shape=list(v.shape),bytes=v.numel()*v.element_size()) for n,v in [('fine',state['value']),('coarse',state['coarse']),('deep_anchor',state['anchor']['value'])]}
            assert sum(v['bytes'] for v in cache.values())==info['cache_bytes']
            if meta['supervised']:
                data=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data);boxes,scores=dataset.post.post_process(data,dict(ego=pred))
                for threshold in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap,threshold)
            record=dict(frame=meta['frame'],supervised=meta['supervised'],refresh=info['late_refresh'],core_conv_flops=cost,cache=cache,calls=calls)
            records.append(record);log.write(json.dumps(record)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='profile',scans=i+1,total=256))
    assert len(records)==256 and model_digest(model)==before
    write_json(out/'ap_records.json',ap)
    assert json.loads((out/'ap_records.json').read_text())==json.loads((source/'final_ap_records.json').read_text())['strict']
    groups={}
    for p in parameter_rows:groups.setdefault(p['category'],dict(parameters=0,core_conv_flops_sum=0,calls=0))['parameters']+=p['elements']
    for record in records:
        for c in record['calls']:
            group=groups[c['category']];group['core_conv_flops_sum']+=c['core_conv_flops'];group['calls']+=1
    total=sum(r['core_conv_flops'] for r in records)
    for group in groups.values():
        group['core_conv_flops_mean']=group['core_conv_flops_sum']/256;group['core_conv_share_percent']=100*group['core_conv_flops_sum']/total
    metrics=dict(parameters=32756052,core_conv_flops_mean=total/256,model_step_ms_mean=None,
        ap50=eval_utils.calculate_ap(ap,.5)[0],ap70=eval_utils.calculate_ap(ap,.7)[0],cache_bytes=info['cache_bytes'])
    for key in metrics:assert metrics[key]==previous['final']['metrics']['strict'][key]
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint_sha256=contract['checkpoint_sha256'],scans=256,labels=sum(r['supervised'] for r in records),
        metrics=metrics,groups=groups,model_weights_and_bn_unchanged=True,reference_frame_cost_refresh_equal=True,reference_raw_ap_equal=True,
        trained=False,timing_valid=False,boundary='Train subset operator audit, not full-val ranking. Activation sparsity alone is not sparse hardware execution or SNN value.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','reference','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min audit budget')));signal.alarm(1800)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<20000:raise RuntimeError('Requires20GB free on one idle GPU')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.audit_int_pyramid_operators']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
