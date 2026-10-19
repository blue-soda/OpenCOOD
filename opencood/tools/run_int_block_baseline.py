"""Audit the pretrained block-cache bridge; optionally evaluate its dense path.

CPU engineering is not an AP or latency result. GPU full validation is guarded
by an idle-device check. No partial-cache accuracy sweep or training is launched.
"""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import torch
from opencood.models.sub_modules.int_block_cache import BlockCacheINT
from opencood.tools.run_int_cache_repair import setup, evaluate
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest


def audit(a, model, cfg, rows, out):
    bridge = BlockCacheINT(model, cfg)
    before_keys = tuple(model.state_dict())
    state_reference = state_bridge = None
    records = []
    macs = {}
    hooks = []
    def count(name):
        def callback(module, inputs, output):
            if isinstance(module, torch.nn.ConvTranspose2d):
                value = inputs[0].numel() * (module.out_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            else:
                value = output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            group = ('spatial_blocks' if name.startswith('backbone.resnet.') else
                     'decoder' if name.startswith('backbone.deblocks.') else name.split('.')[0])
            macs[group] = macs.get(group, 0) + value
        return callback
    _, stream = loader(rows[:a.frames], cfg, 0, 303)
    with torch.no_grad():
        for index, sample in enumerate(stream):
            meta = sample['meta']
            ego = to_device(sample['ego'], torch.device(a.device))
            expected, state_reference, _ = model.step(ego, state_reference, meta, reset=meta.get('reset_before', False))
            if index == 0:
                hooks = [module.register_forward_hook(count(name)) for name, module in model.named_modules()
                         if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
            try:
                actual, state_bridge, info = bridge.step(ego, state_bridge, meta)
            finally:
                for hook in hooks:
                    hook.remove()
                hooks = []
            errors = {k: float((actual[k] - expected[k]).abs().max()) for k in expected}
            assert all(torch.isfinite(v).all() for v in actual.values())
            assert all(torch.allclose(actual[k], expected[k], atol=1e-5, rtol=1e-5) for k in expected), errors
            assert torch.equal(state_reference['value'], state_bridge['fm']['value'])
            assert len(state_bridge['blocks']) == len(bridge.blocks.contract)
            assert all(s['actual_write_fraction'] == 1 for s in info['cache_stages'])
            records.append(dict(frame=meta['frame'],segment=meta['segment'],reset=info['reset'],
                                prediction_max_errors=errors,cache_bytes=info['cache_bytes'],
                                block_shapes=[list(v.shape) for v in state_bridge['blocks']]))
            write_json(out/'status.json',dict(status='RUNNING',phase='engineering',frames=len(records),total=a.frames))
            print('ENGINEERING', json.dumps(records[-1]), flush=True)
    assert tuple(model.state_dict()) == before_keys
    return dict(status='PASSED',device=a.device,frames=records,block_contract=bridge.blocks.contract,
                additional_parameters=0,parameters=sum(p.numel() for p in model.parameters()),
                first_frame_conv_macs=macs,
                cost_scope='Analytic Conv2d/ConvTranspose2d MACs of one real dense frame only; excludes Pillar Linear, BN, activations, warp, masks, copies; not actual latency or total network work.',
                scope='Original-weight inference bridge: exact dense operation order, no learned architecture change.',
                caveats=['Partial mode is experimental and uncalibrated, with no AP or speed claim.',
                         'Pillar, GRU, decoder, shrink and heads still execute densely.',
                         'All residual-block caches consume extra memory and warp/index costs during partial execution.'])


def worker(a):
    out = Path(a.output)
    for key, value in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0,max_age=.25,
                           oracle_stage0=False,oracle_index=0,support_profile=False,zero_uncovered=False).items():
        setattr(a,key,value)
    write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    model, cfg, rows, provenance = setup(a)
    write_json(out/'provenance.json', provenance)
    result = subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError('INT tests failed')
    engineering = audit(a, model, cfg, rows, out)
    write_json(out/'engineering.json',engineering)
    if a.full:
        a.stage = 'full'
        a.engineering = str(out/'engineering.json')
        write_json(out/'status.json',dict(status='RUNNING',phase='full_validation'))
        torch.cuda.reset_peak_memory_stats()
        report = evaluate(a,model,cfg,rows,out,BlockCacheINT(model,cfg))
        report.update(scope='Pretrained INT block-cache bridge, all regions updated; not sparse inference',
                      source_checkpoint_sha256=provenance['checkpoint']['sha256'],
                      full_update=True,additional_parameters=0)
    else:
        report = dict(status='ENGINEERING_PASSED',validation_ap=None,
                      scope='CPU/GPU engineering only, not full validation or latency evidence',
                      source_checkpoint_sha256=provenance['checkpoint']['sha256'])
    # setup pins a best checkpoint; refuse a result if the source file changed.
    assert digest(provenance['checkpoint']['checkpoint']) == provenance['checkpoint']['sha256']
    write_json(out/'report.json',report)
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--frames',type=int,default=4)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--worker',action='store_true')
    a = parser.parse_args()
    if a.frames < 2 or a.frames > 32:
        parser.error('Engineering requires 2..32 frames')
    if a.full and a.device != 'cuda':
        parser.error('Full timing/validation uses CUDA; CPU is engineering only')
    if a.device == 'cuda' and a.gpu is None:
        parser.error('CUDA requires an explicit GPU')
    out = Path(a.output)
    if a.worker:
        def expired(signum, frame):
            raise TimeoutError('Block baseline exceeded its recorded execution budget')
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(7200 if a.full else 1800)
        try:
            worker(a)
        except Exception as exc:
            write_json(out/'status.json',dict(status='FAILED',error=repr(exc)))
            raise
        finally:
            signal.alarm(0)
        return
    if a.device == 'cuda':
        free, util = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
            '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
        if int(free) < 22000 or int(util) > 5:
            raise RuntimeError('GPU is occupied; do not share without explicit scheduling')
    out.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES='' if a.device=='cpu' else str(a.gpu),
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command = [sys.executable,'-m','opencood.tools.run_int_block_baseline'] + sys.argv[1:] + ['--worker']
    with (out/'run.log').open('x') as log:
        process = subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=process.pid,command=command,created_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        device=a.device,timeout_seconds=1800 if not a.full else 7200))
    print(json.dumps(dict(pid=process.pid,output=str(out))))


if __name__ == '__main__':
    main()
