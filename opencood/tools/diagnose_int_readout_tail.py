"""Frozen training diagnostic: analog exceptions isolate quantization error."""
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
from opencood.models.sub_modules.int_if_activation import readout_coded_anchor
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    source=Path(a.source);old=json.loads(source.read_text());scale_file=source.parent/'scales.json'
    assert old['status']=='COMPLETED' and old['scans']==256 and old['split']=='train'
    assert old['checkpoint_sha256']==digest(Path(a.checkpoint)) and old['scale_sha256']==digest(scale_file)
    scales=json.loads(scale_file.read_text())['scales']
    gold=[json.loads(x) for x in (source.parent/'frames.jsonl').read_text().splitlines()]
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    write_json(out/'contract.json',dict(split='train',frames=256,checkpoint_sha256=digest(Path(a.checkpoint)),source_sha256=digest(source),
        scale_sha256=digest(scale_file),scales=scales,steps=8,quantum=65536,
        candidates=['ann','count8','clip_tail','count8_tail','if8_tail'],
        bypass='At each site, positive activation strictly greater than original scale keeps its full analog value and has zero spike drive; other elements use original Q16 coding. Equality to scale stays coded.',
        policy='Original frozen strict cache, both readout sites, no cross-frame membrane. No scale, T, weight or AP search.',
        accounting='Analog value payload=exception count*dtype bytes, conceptual values only, excludes indices/packing; implementation is dense where with full mask. Mask bytes counted per call, not allocator peak. All IF elements and all Conv still execute.',
        gates='Full INT tests; per-frame clip_tail=ANN and count8_tail=if8_tail predictions exact; cache/Conv identity before AP; first32 gate saved.',
        trained=False,timing_valid=False))
    models=dict(ann=AnchoredLatePyramidDetector(base),count8=readout_coded_anchor(base,scales,'count'))
    for name,mode in [('clip_tail','clip'),('count8_tail','count'),('if8_tail','if')]:models[name]=readout_coded_anchor(base,scales,mode,8,True)
    states={k:None for k in models};base_state=None;records=[];ap={k:{.7:dict(tp=[],fp=[],score=[],gt=0)} for k in models}
    params={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()};assert len(set(params.values()))==1
    dataset,stream=loader(rows[:256],cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});preds={}
            for name,model in models.items():
                (pred,states[name],info),cost=measured_step(model,x,states[name],meta);preds[name]=pred
                assert all(torch.isfinite(v).all() for v in pred.values())
                assert meta['frame']==gold[index]['frame'] and cost==gold[index]['arms']['ann']['core_flops']
                assert info['late_refresh']==gold[index]['arms']['ann']['refresh']
                assert all(torch.equal(states[name][k],states['ann'][k]) for k in ('value','coarse'))
                assert torch.equal(states[name]['anchor']['value'],states['ann']['anchor']['value'])
                coding=None if name=='ann' else model.core.readout.coding_stats()
                if name!='ann':assert model.core.readout.input_codec.calls==model.core.readout.middle_codec.calls==index+1
                entry['arms'][name]=dict(core_flops=cost,refresh=info['late_refresh'],coding=coding,
                    relative_mse={k:float((pred[k]-preds['ann'][k]).square().mean()/(preds['ann'][k].square().mean()+1e-12)) for k in pred})
            assert all(torch.equal(preds['ann'][k],preds['clip_tail'][k]) for k in preds['ann'])
            assert all(torch.equal(preds['count8_tail'][k],preds['if8_tail'][k]) for k in preds['ann'])
            entry.update(clip_tail_exact_ann=True,if_count_exact=True,all_cache_states_equal_ann=True)
            if meta['supervised']:
                data=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data)
                for name,pred in preds.items():
                    boxes,scores=dataset.post.post_process(data,dict(ego=pred));eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],.7)
            records.append(entry);log.write(json.dumps(entry)+'\n')
            if index==31:write_json(out/'engineering.json',dict(status='PASSED',frames=32,clip_tail_exact_ann=True,if_count_exact=True,all_cache_states_and_core_equal_ann=True))
            if (index+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='tail_diagnostic',scans=index+1))
    assert len(records)==256
    write_json(out/'ap_records.json',ap);old_ap=json.loads((source.parent/'ap_records.json').read_text())
    for name in ('ann','count8'):assert ap[name][.7]==old_ap[name]['0.7']
    assert ap['clip_tail']==ap['ann'] and ap['count8_tail']==ap['if8_tail']
    metrics={}
    for name in models:
        sites={}
        if name!='ann':
            for site in ('input','middle'):
                values=[r['arms'][name]['coding'][site] for r in records];elements=sum(v['elements'] for v in values)
                sites[site]=dict(elements=elements,analog_bypass_elements=sum(v['analog_bypass_elements'] for v in values),
                    analog_bypass_fraction=sum(v['analog_bypass_elements'] for v in values)/elements,
                    cumulative_analog_value_payload_bytes=sum(v['analog_value_payload_bytes'] for v in values),
                    max_single_call_bypass_mask_bytes=max(v['bypass_mask_bytes'] for v in values),
                    explicit_neuron_updates=sum(v['if_neuron_updates'] for v in values),
                    emission_rate=None if values[0]['spike_count'] is None else sum(v['spike_count'] for v in values)/(8*elements))
        metrics[name]=dict(ap70=eval_utils.calculate_ap(ap[name],.7)[0],sites=sites,
            core_flops=sum(r['arms'][name]['core_flops'] for r in records),
            mean_relative_mse={k:sum(r['arms'][name]['relative_mse'][k] for r in records)/256 for k in preds[name]})
    write_json(out/'report.json',dict(status='COMPLETED',split='train',scans=256,labels=sum(r['supervised'] for r in records),
        checkpoint_sha256=digest(Path(a.checkpoint)),scale_sha256=digest(scale_file),parameters=params,metrics=metrics,
        clip_tail_exact_ann_all=True,if_count_exact_all=True,all_cache_states_and_core_equal_ann=True,
        ann_and_count8_raw_ap_equal_R085=True,scale_buffer_bytes=2560,persistent_membrane_bytes=0,timing_valid=False,
        caveat='Train-prefix hybrid analog exceptions, not pure SNN. Payload values exclude positions and packing; implementation uses dense masks and where. Conv and full-grid IF updates unchanged; no execution or energy savings.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','source'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h tail-bypass diagnostic budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('At least12GB free required')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.diagnose_int_readout_tail']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
