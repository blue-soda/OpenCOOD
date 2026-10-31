"""Frozen full-stream evaluation of the preselected period-two strict anchor."""
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
from opencood.tools.run_int_cache_repair import setup,evaluate
from opencood.tools.evaluate_int_block_cache import MeasuredCache
from opencood.tools.run_int_ego import write_json
from opencood.tools.launch_int_memory import digest


class MeasuredAnchor(MeasuredCache):
    def step(self,ego,state,meta):
        self.core=self.pillar=0
        pred,state,info=self.model.step(ego,state,meta)
        info.update(core_conv_flops=self.core,pillar_linear_flops=self.pillar)
        self.records.append(dict(core=self.core,pillar=self.pillar,state_bytes=info['state_bytes'],
            cache_bytes=info['cache_bytes'],refresh=info['late_refresh'],uncovered=info['uncovered_fraction']))
        return pred,state,info


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    # This frozen inference fits within the observed single-model envelope.
    # Bound this worker's allocator when sharing; CUDA context is additional.
    torch.cuda.set_per_process_memory_fraction(.20)
    source=Path(a.diagnostic_report);gate=json.loads(source.read_text())
    assert gate['status']=='COMPLETED' and gate['split']=='train' and gate['scans']==256
    assert gate['dense_period1_exact_all'] and gate['checkpoint_sha256']==digest(Path(a.checkpoint))
    source_contract=json.loads((source.parent/'contract.json').read_text())
    assert source_contract['candidates']['period2_strict']==dict(period=2,coverage='strict')
    assert source_contract['max_age']==.25
    tests=(source.parent/'unit_tests.log').read_text()
    assert 'Ran 293 tests' in tests and tests.rstrip().endswith('OK')
    with (source.parent/'frames.jsonl').open() as stream:dense=json.loads(next(stream))['arms']['dense']['core_conv_flops']
    for key,value in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0,max_age=.25,
        oracle_stage0=False,oracle_index=0,support_profile=False,zero_uncovered=False).items():setattr(a,key,value)
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    model=AnchoredLatePyramidDetector(base,period=2,max_age=.25,coverage='strict')
    a.engineering=str(out/'engineering.json')
    write_json(a.engineering,dict(status='PASSED',source_report=str(source),source_sha256=digest(source),
        scope='Reuses unchanged anchor implementation gate: 293 tests and 256 training frames, period1 exact'))
    write_json(out/'policy.json',dict(period=2,coverage='strict',max_age=.25,selection_split='train',
        source_diagnostic_sha256=digest(source),checkpoint_sha256=gate['checkpoint_sha256'],
        trained=False,allocator_fraction=.20,timing_valid=False,
        boundary='Full validation exploratory; no zero-fill approximation on reuse, no tuning on validation'))
    measured=MeasuredAnchor(model);a.stage='full'
    write_json(out/'status.json',dict(status='RUNNING',phase='full_validation'))
    try:report=evaluate(a,teacher,cfg,rows,out,measured)
    finally:measured.close()
    n=len(measured.records)
    report.update(scope='Period2 strict fixed-anchor full validation',period=2,coverage='strict',max_age=.25,
        checkpoint_sha256=gate['checkpoint_sha256'],timing_valid=False,
        core_conv_flops_mean=sum(r['core'] for r in measured.records)/n,
        core_conv_ratio=sum(r['core'] for r in measured.records)/(n*dense),dense_core_conv_flops=dense,
        pillar_linear_flops_mean=sum(r['pillar'] for r in measured.records)/n,
        reuse_frames=sum(r['refresh'] is None for r in measured.records),
        late_refresh_counts={str(reason):sum(r['refresh']==reason for r in measured.records)
            for reason in sorted({r['refresh'] for r in measured.records},key=str)},
        state_bytes=measured.records[-1]['state_bytes'],cache_bytes=measured.records[-1]['cache_bytes'])
    for key in list(report):
        if key.startswith('model_step_ms'):report[key]=None
    report['caveats']=['Policy selected on training prefix, full validation exploratory not independent test.',
        'Fine/coarse/decoder/heads remain dense; whole deep encoder/update skipped only on eligible reuse frames.',
        'Core convolution and Pillar Linear work measured separately; warp, nonlinear ops, memory movement and IO excluded from those counts.',
        'Shared GPU and count hooks invalidate latency; independent exclusive no-hook timing required.',
        'Persistent fixed anchor changes deep recurrent history semantics; no SNN or energy claim.']
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','diagnostic-report'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('2h anchor evaluation budget')));signal.alarm(7200)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<6000:raise RuntimeError('Frozen evaluation requires 6000 MiB free; no contention timing claims')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.evaluate_int_late_anchor']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,free_mib_before=free,allocator_fraction=.20,
        command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
