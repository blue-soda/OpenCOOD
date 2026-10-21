"""Frozen transfer of mature stage repairs into a compact spatial readout.

This is a full-image repair control, before selective regional execution.
All checkpoints are fixed and verified; no optimization happens here.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_compact_readout import compact_shrink, cached_features
from opencood.models.sub_modules.int_light_repair import make_repairs
from opencood.models.sub_modules.int_local_cache import LocalCacheDetector
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def completed_best(directory):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    assert json.loads((directory/'status.json').read_text())['status'] == 'COMPLETED'
    assert report['stop_reason'] == 'plateau'
    best = json.loads((directory/'best.json').read_text())
    assert digest(best['checkpoint']) == best['sha256']
    return best, torch.load(best['checkpoint'], map_location='cpu')


def worker(a):
    out = Path(a.output)
    write_json(out/'status.json', dict(status='RUNNING', phase='setup'))
    for k,v in dict(mode='gru', stage='pilot', device='cuda', content_calibration=None,
                    periodic_last=0, max_age=.25).items():
        setattr(a,k,v)
    original,cfg,rows,provenance = setup(a)
    readout_best,readout = completed_best(a.readout_run)
    repair_best,repair = completed_best(a.repair_run)
    source = provenance['checkpoint']['sha256']
    assert readout['contract']['source_sha256'] == source
    assert readout['contract']['data_contract'] == provenance['contract']
    assert repair['contract']['teacher_checkpoint_sha256'] == source
    assert repair['contract']['repair_period'] == 3
    assert repair['contract']['repair_policy'] == 'dense'
    assert repair['contract'].get('repair_target','stages') == 'stages'
    repair_provenance = json.loads((Path(a.repair_run)/'provenance.json').read_text())
    assert repair_provenance['contract'] == provenance['contract']
    student = copy.deepcopy(original)
    student.shrink_conv = compact_shrink(original.shrink_conv,readout['contract']['rank'])[0].cuda()
    student.shrink_conv.load_state_dict(readout['readout'],strict=True)
    student.eval()
    repairs = make_repairs().cuda().eval()
    repairs.load_state_dict(repair['repairs'],strict=True)
    for module in (original,student,repairs):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    modes = dict(original=original, compact_full=student,
        compact_cache=LocalCacheDetector(student,cfg,0.,period=3,repair_modules=cached_features(),repair_dense=True),
        original_repair=LocalCacheDetector(original,cfg,0.,period=3,repair_modules=repairs,repair_dense=True),
        compact_repair=LocalCacheDetector(student,cfg,0.,period=3,repair_modules=repairs,repair_dense=True))
    expected = dict(original=.758736104616861,compact_cache=readout_best['ap70'],original_repair=repair_best['ap70'])
    write_json(out/'provenance.json',dict(source=provenance,readout=readout_best,repair=repair_best,
        rank=readout['contract']['rank'],protocol='Frozen transfer; period3 immutable stage2/3 anchors, current E and compact readout every frame; full-image repairs only.',
        expected_ap70=expected))
    test = subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(test.stdout)
    assert test.returncode == 0
    costs = {};states = {k:None for k in modes};partials = 0;captured = {};calls = [0,0]
    def count(i):
        def hook(m,x,y): calls[i] += 1
        return hook
    hooks = [original.shrink_conv.register_forward_pre_hook(lambda m,x:captured.update(original=x[0])),
             student.shrink_conv.register_forward_pre_hook(lambda m,x:captured.update(student=x[0]))]
    hooks += [getattr(student.backbone.resnet,'layer%d'%i).register_forward_hook(count(i-1)) for i in (1,2)]
    _,stream = loader(rows[:32],cfg,0,303)
    try:
        with torch.no_grad():
            for sample in stream:
                ego = to_device(sample['ego'],torch.device('cuda'));meta = sample['meta']
                full_pred = None
                for name,mode in modes.items():
                    model = original if name in ('original','original_repair') else student
                    counter = ConvCounter(model,repairs if name.endswith('_repair') else torch.nn.ModuleList())
                    previous = states[name];before = list(calls)
                    try: pred,states[name],info = mode.step(ego,states[name],meta)
                    finally: counter.close()
                    flag = bool(info.get('local_partial'))
                    key = name+('_partial' if flag else '_full')
                    if key in costs: assert costs[key] == counter.total
                    costs[key] = counter.total
                    if name == 'compact_full': full_pred = pred
                    if name == 'compact_repair':
                        assert torch.equal(captured['original'],captured['student']), 'Repair readout inputs differ'
                        if flag:
                            partials += 1
                            assert calls == before, 'Partial repair executed original deep stages'
                            assert all(x is y for x,y in zip(previous['cache']['values'],states[name]['cache']['values']))
                        else:
                            assert all(torch.equal(pred[k],full_pred[k]) for k in pred)
    finally:
        for hook in hooks: hook.remove()
    assert partials > 0
    write_json(out/'engineering.json',dict(status='PASSED',frames=32,partial_frames=partials,
        shared_repaired_inputs_exact=True,immutable_anchors=True,partial_deep_calls=0,conv_macs=costs))
    states = {k:None for k in modes}
    stats = {k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    times = {k:[] for k in modes};compute = {k:0 for k in modes};partial = {k:0 for k in modes}
    dataset,stream = loader(rows,cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego = to_device(sample['ego'],torch.device('cuda'));meta = sample['meta']
            names = list(modes);offset = index%len(names);names = names[offset:]+names[:offset]
            for name in names:
                torch.cuda.synchronize();tick = time.perf_counter()
                pred,states[name],info = modes[name].step(ego,states[name],meta)
                torch.cuda.synchronize();ms = 1000*(time.perf_counter()-tick)
                assert all(torch.isfinite(x).all() for x in pred.values())
                if index >= 10: times[name].append(ms)
                flag = bool(info.get('local_partial'));partial[name] += flag
                compute[name] += costs[name+('_partial' if flag else '_full')]
                if meta['supervised']:
                    data = dict(ego=ego)
                    boxes,scores = dataset.post.post_process(data,dict(ego=pred))
                    gt = dataset.post.generate_gt_bbx(data)
                    for t in stats[name]: eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
            if (index+1)%200 == 0:
                write_json(out/'status.json',dict(status='RUNNING',phase='validation',scans=index+1))
    assert index+1 == len(rows)
    results = {k:dict(ap70=eval_utils.calculate_ap(stats[k],.7)[0],ap50=eval_utils.calculate_ap(stats[k],.5)[0],
        mean_ms=float(np.mean(times[k])),p95_ms=float(np.percentile(times[k],95)),
        conv_gflops=2*compute[k]/len(rows)/1e9,partial_frames=partial[k]) for k in modes}
    for k,ap in expected.items(): assert abs(results[k]['ap70']-ap)<1e-8, 'Reloaded AP mismatch: '+k
    assert len({partial[k] for k in ('compact_cache','original_repair','compact_repair')}) == 1
    write_json(out/'report.json',dict(status='COMPLETED',results=results,scans=len(rows),labels=sum(r['supervised'] for r in rows),
        scope='Same-GPU rotating synchronized model time excluding IO/H2D/postprocess. Conv2d/ConvTranspose2d MACs x2 only; excludes VFE Linear, warps, BN, activations. Whole-image small repair; no regional execution or training.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p = argparse.ArgumentParser()
    for name in ('root','readout-run','repair-run','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true')
    a = p.parse_args();out = Path(a.output)
    if a.worker:
        def expired(s,f): raise TimeoutError('Evaluation exceeded 2 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(7200)
        try: worker(a)
        except Exception as exc: write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally: signal.alarm(0)
        return
    free,util = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5: raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command = [sys.executable,'-m','opencood.tools.evaluate_int_compact_repair']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        proc = subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=7200,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__ == '__main__': main()
