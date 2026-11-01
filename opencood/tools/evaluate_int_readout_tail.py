"""One frozen full-validation candidate: centered count plus analog exceptions."""
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
from opencood.tools.evaluate_int_late_anchor import MeasuredAnchor
from opencood.tools.run_int_cache_repair import setup,evaluate
from opencood.tools.run_int_ego import write_json
from opencood.tools.launch_int_memory import digest


class MeasuredReadout(MeasuredAnchor):
    def __init__(self,model,reference,gold,out):
        super().__init__(model);self.reference=reference;self.gold=gold;self.out=out

    def step(self,ego,state,meta):
        index=len(self.records);captured=[];previous=state
        hook=self.model.core.observation.register_forward_pre_hook(lambda m,args:captured.append(args[0])) if index<32 else None
        try:pred,state,info=super().step(ego,state,meta)
        finally:
            if hook:hook.remove()
        gold=self.gold[index]
        assert meta['frame']==gold['frame'] and meta['segment']==gold['segment']
        assert info['core_conv_flops']==gold['core_conv_flops'] and info['late_refresh']==gold['late_refresh']
        assert info['anchor_timestamp_us']==gold['anchor_timestamp_us'] and info['cache_bytes']==gold['cache_bytes']
        if index<32:
            other,other_state,_=self.reference.step_spatial(captured[0],previous,meta)
            assert all(torch.equal(pred[k],other[k]) for k in pred)
            assert all(torch.equal(state[k],other_state[k]) for k in ('value','coarse'))
            assert torch.equal(state['anchor']['value'],other_state['anchor']['value'])
        assert self.model.core.readout.input_codec.calls==self.model.core.readout.middle_codec.calls==index+1
        coding=self.model.core.readout.coding_stats();info['activation_coding']=coding
        info['validation_if_count_exact']=True if index<32 else None
        self.records[-1]['coding']=coding
        if (index+1)%200==0 or index==31:write_json(self.out/'status.json',dict(status='RUNNING',phase='full_validation',scans=index+1,total=5794))
        return pred,state,info


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    torch.cuda.set_per_process_memory_fraction(.5)
    source=Path(a.diagnostic_report);gate=json.loads(source.read_text());contract=json.loads((source.parent/'contract.json').read_text())
    assert gate['status']=='COMPLETED' and gate['split']=='train' and gate['scans']==256 and gate['centered']
    assert gate['centered_if_count_exact_all'] and gate['all_cache_states_and_core_equal_ann']
    assert gate['checkpoint_sha256']==digest(Path(a.checkpoint)) and contract['steps']==8 and contract['quantum']==65536
    assert contract['centered'] and contract['scale_sha256']==gate['scale_sha256']
    tests=(source.parent/'unit_tests.log').read_text();assert 'Ran 317 tests' in tests and tests.rstrip().endswith('OK')
    baseline=Path(a.strict_report);prior=json.loads(baseline.read_text())
    assert prior['status']=='COMPLETED' and prior['scans']==5794 and prior['checkpoint_sha256']==gate['checkpoint_sha256']
    assert prior['coverage']=='strict' and prior['period']==2 and prior['max_age']==.25
    gold=[json.loads(x) for x in (baseline.parent/'frames.jsonl').read_text().splitlines()];assert len(gold)==5794
    for key,value in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0,max_age=.25,
        oracle_stage0=False,oracle_index=0,support_profile=False,zero_uncovered=False).items():setattr(a,key,value)
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    model=readout_coded_anchor(base,contract['scales'],'count',8,True,32768)
    reference=readout_coded_anchor(base,contract['scales'],'if',8,True,32768)
    a.engineering=str(out/'engineering.json')
    write_json(a.engineering,dict(status='PASSED',source_report=str(source),source_sha256=digest(source),
        scope='317 tests and256 train-frame IF/count/cache/Conv parity; first32 val frames additionally compare explicit IF predictions and states.'))
    write_json(out/'policy.json',dict(checkpoint_sha256=gate['checkpoint_sha256'],scale_sha256=gate['scale_sha256'],
        source_diagnostic_sha256=digest(source),strict_report_sha256=digest(baseline),scales=contract['scales'],steps=8,
        initial_membrane_units=32768,quantum=65536,period=2,coverage='strict',max_age=.25,
        candidate='Centered closed-form count with analog exceptions at both readout inputs',trained=False,timing_valid=False,
        boundary='One candidate frozen before full validation; no scale/T/phase search. Count execution plus analog bypass, not sparse IF kernel. Instrumentation and first32 reference work invalidate timing and deployment peak-memory claims.'))
    measured=MeasuredReadout(model,reference,gold,out);a.stage='full'
    try:report=evaluate(a,teacher,cfg,rows,out,measured)
    finally:measured.close()
    assert len(measured.records)==5794 and report['labels']==1738
    sites={}
    for site in ('input','middle'):
        values=[r['coding'][site] for r in measured.records];elements=sum(v['elements'] for v in values)
        sites[site]=dict(elements=elements,analog_bypass_elements=sum(v['analog_bypass_elements'] for v in values),
            analog_bypass_fraction=sum(v['analog_bypass_elements'] for v in values)/elements,
            cumulative_analog_value_payload_bytes=sum(v['analog_value_payload_bytes'] for v in values),
            max_single_call_bypass_mask_bytes=max(v['bypass_mask_bytes'] for v in values),
            count_execution_if_updates=sum(v['if_neuron_updates'] for v in values),
            equivalent_full_explicit_if_updates=8*elements,emission_rate=sum(v['spike_count'] for v in values)/(8*elements))
    report.update(scope='Frozen centered count+analog-tail readout full validation',checkpoint_sha256=gate['checkpoint_sha256'],
        scale_sha256=gate['scale_sha256'],timing_valid=False,core_and_refresh_identity_to_strict_all=True,
        validation_if_count_exact_frames=32,sites=sites,strict_baseline_ap70=prior['metrics']['bev_ap70'],
        ap70_delta_to_strict_pp=100*(report['metrics']['bev_ap70']-prior['metrics']['bev_ap70']),
        core_conv_ratio=prior['core_conv_ratio'],
        core_conv_flops_mean=prior['core_conv_flops_mean'],
        pillar_linear_flops_mean=prior['pillar_linear_flops_mean'],
        parameters=sum(p.numel() for p in model.parameters()),
        timing_invalid_reason='Hooks and first32 explicit IF reference work; isolated timing required',
        scale_buffer_bytes=2560,persistent_membrane_bytes=0)
    for key in list(report):
        if key.startswith('model_step_ms'):report[key]=None
    report['caveats']=['Exploratory full validation with validation-selected seed303 R051; not independent testing or statistical superiority.',
        'Count code is equivalent to explicit IF on training256 plus validation32 frames; remaining full stream uses closed-form count.',
        'Analog exceptions are explicit hybrid payload, not pure SNN. Dense Conv unchanged relative to strict ANN; existing cache saving cannot be attributed to coding.',
        'Payload counts values only, excludes positions/packing; dense mask and where are implemented. Count arithmetic and nonconv costs excluded from Conv counts.',
        'Timing and deployment peak memory invalid due to hooks and first32 IF reference work. No speed or energy claim.']
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','diagnostic-report','strict-report'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h frozen full-validation budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('At least12GB free required')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.evaluate_int_readout_tail']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
