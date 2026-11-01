"""Frozen train256 FP32 scale folding and time-block accumulation diagnostic."""
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
from opencood.models.sub_modules.int_if_activation import readout_coded_anchor
from opencood.models.sub_modules.int_folded_readout import folded_readout,folded_stats
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
    if tests.returncode:raise RuntimeError('INT unit tests failed')
    torch.cuda.set_per_process_memory_fraction(.45)
    source=Path(a.source);prior=json.loads(source.read_text());contract=json.loads((source.parent/'contract.json').read_text())
    assert prior['status']=='COMPLETED' and prior['split']=='train' and prior['scans']==256 and prior['centered']
    assert prior['checkpoint_sha256']==digest(Path(a.checkpoint))
    assert contract['steps']==8 and contract['quantum']==65536 and contract['centered']
    gold=[json.loads(x) for x in (source.parent/'frames.jsonl').read_text().splitlines()]
    full_source=Path(a.validation_reference) if a.validation_reference else None
    if full_source:
        full_prior=json.loads(full_source.read_text())
        assert full_prior['status']=='COMPLETED' and full_prior['scans']==5794 and full_prior['labels']==1738
        assert full_prior['checkpoint_sha256']==prior['checkpoint_sha256'] and full_prior['scale_sha256']==prior['scale_sha256']
        training_gate=Path(a.training_gate);training_report=json.loads(training_gate.read_text())
        assert training_report['status']=='COMPLETED' and training_report['split']=='train' and training_report['scans']==256
        assert training_report['checkpoint_sha256']==prior['checkpoint_sha256'] and training_report['scale_sha256']==prior['scale_sha256']
        assert training_report['all_cache_states_equal'] and training_report['reference_ap70_records_equal_R088']
        expected={'reference','folded_count','event8'} | ({'folded_count_int8','event8_int8'} if a.int8_weights else set())
        assert set(training_report['metrics'])==expected
        if a.int8_weights:
            assert training_report['fp32_control_ap_records_equal']
            training_contract=json.loads((training_gate.parent/'contract.json').read_text())
            assert training_contract['weight_bits']==8 and training_contract['steps']==8 and training_contract['initial_membrane_units']==32768
        gold=[json.loads(x) for x in (full_source.parent/'frames.jsonl').read_text().splitlines()]
    frames=5794 if full_source else 256
    split='val' if full_source else 'train'
    a.mode='gru';a.stage='pilot' if full_source else 'calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    names=['reference','folded_count','event8']+(['folded_count_int8','event8_int8'] if a.int8_weights else [])
    models={name:readout_coded_anchor(base,contract['scales'],'count',8,True,32768) for name in names}
    for name in names[1:]:
        mode=name.replace('_int8','')
        models[name].core.readout=folded_readout(models[name].core.readout,mode,8 if name.endswith('_int8') else None)
    quantization={}
    if a.int8_weights:
        fp32_source=Path(a.fp32_reference);fp32_prior=json.loads(fp32_source.read_text())
        assert fp32_prior['status']=='COMPLETED' and fp32_prior['split']==split and fp32_prior['scans']==frames
        assert fp32_prior['checkpoint_sha256']==prior['checkpoint_sha256'] and fp32_prior['scale_sha256']==prior['scale_sha256']
        for name in names[-2:]:
            block=models[name].core.readout.body.layers[0].double_conv
            quantization[name]={site:block[index].quantization for site,index in [('input',0),('middle',3)]}
        assert quantization['folded_count_int8']==quantization['event8_int8']
        if full_source:assert quantization==training_report['quantization']==training_contract['quantization']
    params={name:sum(p.numel() for p in model.parameters()) for name,model in models.items()}
    assert len(set(params.values()))==1
    write_json(out/'contract.json',dict(source_sha256=digest(source),checkpoint_sha256=digest(Path(a.checkpoint)),
        scale_sha256=prior['scale_sha256'],split=split,frames=frames,steps=8,quantum=65536,initial_membrane_units=32768,
        validation_reference_sha256=digest(full_source) if full_source else None,
        training_gate_sha256=digest(Path(a.training_gate)) if full_source else None,
        modes=list(models),parameters=params,precision='INT8 storage roundtrip, FP32 dequantized convolution and accumulation; controls FP32; TF32 disabled' if a.int8_weights else 'FP32 weights/convolution outputs/time-block accumulator; TF32 disabled',
        weight_bits=8 if a.int8_weights else None,quantization=quantization,
        fp32_source_sha256=digest(Path(a.fp32_reference)) if a.int8_weights else None,
        actual_model_buffer_bytes={name:sum(b.numel()*b.element_size() for b in model.buffers()) for name,model in models.items()},
        scope='Dense GPU numerical emulation. Folded count uses two Conv per site; event8 uses nine. Bias once, unchanged BN.',
        boundary='Library chooses within-Conv summation. Not per-synapse hardware arithmetic, sparse kernel, speed or energy evidence.',
        trained=False,tuned=False,timing_valid=False))
    states={name:None for name in models};base_state=None;records=[]
    ap={name:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for name in models}
    dataset,stream=loader(rows[:frames],cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            assert meta['frame']==gold[index]['frame']
            entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});preds={}
            for name,model in models.items():
                (pred,states[name],info),hook_flops=measured_step(model,x,states[name],meta);preds[name]=pred
                assert all(torch.isfinite(value).all() for value in pred.values())
                assert info['late_refresh']==(gold[index]['late_refresh'] if full_source else gold[index]['arms']['ann']['refresh'])
                assert all(torch.equal(states[name][k],states['reference'][k]) for k in ('value','coarse'))
                assert torch.equal(states[name]['anchor']['value'],states['reference']['anchor']['value'])
                sites=None if name=='reference' else folded_stats(model.core.readout)
                functional_flops=0 if sites is None else sum(v['functional_conv_flops'] for v in sites.values())
                cost=hook_flops+functional_flops
                if name=='reference':
                    assert cost==(gold[index]['core_conv_flops'] if full_source else gold[index]['arms']['ann']['core_flops'])
                else:
                    readout_flops=sum(v['reference_conv_flops'] for v in sites.values())
                    assert cost==entry['arms']['reference']['core_conv_flops']+readout_flops*(1 if name.startswith('folded_count') else 8)
                entry['arms'][name]=dict(core_conv_flops=cost,functional_conv_flops=functional_flops,sites=sites,
                    max_abs_error={k:float((pred[k]-preds['reference'][k]).abs().max()) for k in pred},
                    relative_mse={k:float((pred[k]-preds['reference'][k]).square().mean()/(preds['reference'][k].square().mean()+1e-12)) for k in pred})
            if meta['supervised']:
                data=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data)
                for name,pred in preds.items():
                    boxes,scores=dataset.post.post_process(data,dict(ego=pred))
                    for threshold in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],threshold)
            records.append(entry);log.write(json.dumps(entry)+'\n')
            if index==31:write_json(out/'engineering.json',dict(status='PASSED',frames=32,cache_states_equal=True,conv_accounting_verified=True))
            if (index+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=split+'_diagnostic',scans=index+1,total=frames))
    assert len(records)==frames
    write_json(out/'ap_records.json',ap)
    old_ap=json.loads(((full_source or source).parent/'ap_records.json').read_text())
    if full_source:
        assert all(ap['reference'][t]==old_ap[str(t)] for t in (.5,.7))
    else:assert ap['reference'][.7]==old_ap['count8_centered']['0.7']
    if a.int8_weights:
        fp32_ap=json.loads((Path(a.fp32_reference).parent/'ap_records.json').read_text())
        assert all(ap[name][t]==fp32_ap[name][str(t)] for name in names[:3] for t in (.5,.7))
    metrics={name:dict(parameters=params[name],ap50=eval_utils.calculate_ap(ap[name],.5)[0],ap70=eval_utils.calculate_ap(ap[name],.7)[0],
        core_conv_flops_mean=sum(r['arms'][name]['core_conv_flops'] for r in records)/frames,
        max_abs_error={k:max(r['arms'][name]['max_abs_error'][k] for r in records) for k in preds[name]},
        mean_relative_mse={k:sum(r['arms'][name]['relative_mse'][k] for r in records)/frames for k in preds[name]},
        model_step_ms_mean=None,timing_invalid_reason='Multi-arm instrumented numerical diagnostic; no valid timing',
        extra_folded_weight_bytes=0 if name=='reference' else sum(v['folded_weight_bytes'] for v in records[0]['arms'][name]['sites'].values())) for name in models}
    write_json(out/'report.json',dict(status='COMPLETED',split=split,scans=frames,labels=sum(r['supervised'] for r in records),
        checkpoint_sha256=digest(Path(a.checkpoint)),scale_sha256=prior['scale_sha256'],source_sha256=digest(source),
        metrics=metrics,quantization=quantization,fp32_control_ap_records_equal=True if a.int8_weights else None,
        reference_ap70_records_equal_R088=True if not full_source else None,
        reference_raw_ap_equal_R089=True if full_source else None,all_cache_states_equal=True,timing_valid=False,
        caveat='FP32 summation-order diagnostic; split explicitly recorded, validation-selected seed303 not independent test. Dense repeated Conv FLOPs include simulation overhead; no sparse hardware claim. Pillar computed once for shared spatial inputs and excluded from per-arm core budgets.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for name in ('root','source','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--validation-reference');p.add_argument('--training-gate')
    p.add_argument('--int8-weights',action='store_true');p.add_argument('--fp32-reference');a=p.parse_args();out=Path(a.output)
    if a.int8_weights and not a.fp32_reference:p.error('INT8 study requires a completed FP32 diagnostic on the same split')
    if a.validation_reference and not a.training_gate:p.error('Full validation requires completed frozen three-arm training diagnostic')
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h frozen FP32 diagnostic budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('At least12GB free required')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.diagnose_int_folded_readout']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
