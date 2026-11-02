"""Diagnostic-free spatial activation parity gate and exclusive model timing.

The target input replacement pre-hook is part of model execution. Activity and
FLOP collection are excluded from timing; controller synchronizations remain.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_spatial_spike import SpatialSpikeAnchor
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.benchmark_int_late_cache import check_exclusive


def exact_tree(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b) and bool(torch.isfinite(b).all())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:exact_tree(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):exact_tree(x, y)
    else:assert a == b


def compare_info(reference, runtime):
    reference = dict(reference);runtime = dict(runtime)
    if 'spatial_activation' in runtime:
        a = reference.pop('spatial_activation');b = runtime.pop('spatial_activation')
        assert b == {k: a[k] for k in b}
        assert 'emission_nonzero_fraction' not in b
    exact_tree(reference, runtime)


def set_statistics(model, enabled):
    if isinstance(model, SpatialSpikeAnchor):model.activation_memory.collect_statistics = enabled


def counted_step(model, ego, state, meta):
    macs = [0]
    def count(m, inputs, output):
        n = inputs[0].numel() * (m.out_channels // m.groups) if isinstance(m, torch.nn.ConvTranspose2d) else output.numel() * (m.in_channels // m.groups)
        macs[0] += n * m.kernel_size[0] * m.kernel_size[1]
    hooks = [m.register_forward_hook(count) for m in model.core.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    try:result = model.step(ego, state, meta)
    finally:
        for h in hooks:h.remove()
    return result, 2 * macs[0]


def worker(a):
    out = Path(a.output);write_json(out/'status.json', dict(status='RUNNING', phase='tests'))
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'opencood/test', '-p', 'test_int*.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    occupancy = [check_exclusive(a.gpu)] if a.suite == 'timing' else []
    source = Path(a.full_reference);prior = json.loads((source/'report.json').read_text())
    source_contract = json.loads((source/'contract.json').read_text())
    pilot = Path(a.pilot);pc = json.loads((pilot/'contract.json').read_text())
    pr = json.loads((pilot/'report.json').read_text())
    gold = [json.loads(x) for x in (source/'full_frames.jsonl').read_text().splitlines()]
    assert prior['status'] == pr['status'] == 'COMPLETED'
    assert prior['all_weights_bn_scale_unchanged'] and prior['strict_full_reference_verified']
    assert prior['checkpoint_sha256'] == digest(Path(a.checkpoint)) == pr['checkpoint_sha256']
    assert len(gold) == prior['evaluation']['scans'] == 5794 and prior['evaluation']['labels'] == 1738
    if a.suite == 'timing':
        gate = json.loads(Path(a.gate_report).read_text())
        gate_contract = json.loads((Path(a.gate_report).parent/'contract.json').read_text())
        source_launch = json.loads((source/'launch.json').read_text())
        assert gate['status'] == 'COMPLETED' and gate['suite'] == 'gate' and gate['exact_frames'] == 64
        assert gate['source_report_sha256'] == digest(source/'report.json')
        # A new host/environment needs its own full AP reference and parity gate.
        # Agreement between two current implementations cannot prove that an
        # older PyTorch environment produced the same detections.
        assert source_launch['command'][0] == sys.executable, 'Re-evaluate full AP in this interpreter before timing'
        assert gate_contract['torch_version'] == torch.__version__
        assert gate_contract['gpu'] == torch.cuda.get_device_name(0)
    a.mode='gru';a.stage='full';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher, cfg, rows, provenance = setup(a);write_json(out/'provenance.json', provenance)
    assert [r['frame'] for r in rows] == [r['frame'] for r in gold]
    base = PyramidStateDetector(teacher, cfg).cuda().eval();saved = torch.load(a.checkpoint, map_location='cpu')
    assert saved['contract']['teacher_sha256'] == provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract'] == provenance['contract'] and saved['contract']['version'] == base.version
    base.load_state_dict(saved['model'], strict=True);models = dict(strict=AnchoredLatePyramidDetector(base));shas = {}
    for name in ('ann', 'lif', 'leaky'):
        path = pilot/(name+'_target.pth');shas[name] = digest(path)
        assert shas[name] == pr['training'][name]['checkpoint_sha256'] == source_contract['target_checkpoint_sha256'][name]
        value = torch.load(str(path), map_location='cpu')
        assert value['mode'] == name and value['updates'] == 71 and value['contract'] == pc
        model = SpatialSpikeAnchor(base, name, pc['scale'], collect_statistics=False).cuda().eval()
        with torch.no_grad():model.core.coarse_encoder[3].conv2.weight.copy_(value['target_weight'])
        assert torch.equal(model.activation_memory.scale.cpu(), value['scale']);models[name] = model
    for m in models.values():
        for p in m.parameters():p.requires_grad_(False)
    before = {k:model_digest(m) for k,m in models.items()}
    names = list(models);orders = [names] if a.suite == 'gate' else [names[i:]+names[:i] for i in range(4)]
    frames = 64 if a.suite == 'gate' else 5794
    contract = dict(suite=a.suite, source_report_sha256=digest(source/'report.json'), source_frames_sha256=digest(source/'full_frames.jsonl'),
        pilot_report_sha256=digest(pilot/'report.json'), checkpoint_sha256=prior['checkpoint_sha256'], target_checkpoint_sha256=shas,
        frames=frames, rounds=len(orders), orders=orders, warmup=16, trained=False, ap_recomputed=False,
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__, precision='FP32, TF32 disabled',
        scope='Synchronized model.step including Pillar/warp/cache/readout/head and target replacement hook; excluding IO/H2D/postprocessing/diagnostic reductions.',
        parity='All nested states, predictions and common metadata exact versus statistics-enabled path in first round; own state per path/arm.',
        finite_checks='Outside timed interval; runtime mode omits diagnostic finite reductions.',
        budget_minutes=30 if a.suite == 'gate' else 90)
    write_json(out/'contract.json', contract)
    timings = {n:[] for n in names};rounds=[];costs={n:[] for n in names}
    with torch.no_grad(), (out/'frames.jsonl').open('x', buffering=1) as log:
        for repeat, order in enumerate(orders):
            states={n:None for n in names};ref_states={n:None for n in names};local={n:[] for n in names}
            _, stream=loader(rows[:frames], cfg, 2, 303)
            for index, sample in enumerate(stream):
                ego=to_device(sample['ego'], torch.device('cuda'));meta=sample['meta']
                assert meta['frame']==gold[index]['frame'] and bool(meta['supervised'])==bool(gold[index]['supervised'])
                predictions={};infos={};elapsed={};cost={}
                for n in order:
                    set_statistics(models[n], False)
                    if a.suite == 'gate':
                        (predictions[n],states[n],infos[n]),cost[n]=counted_step(models[n],ego,states[n],meta)
                        costs[n].append(cost[n]);assert cost[n]==gold[index]['arms'][n]['core_conv_flops']
                    else:
                        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in models[n].modules())
                        torch.cuda.synchronize();tick=time.perf_counter()
                        predictions[n],states[n],infos[n]=models[n].step(ego,states[n],meta)
                        torch.cuda.synchronize();elapsed[n]=1000*(time.perf_counter()-tick)
                        if index>=16:local[n].append(elapsed[n]);timings[n].append(elapsed[n])
                for n in names:
                    assert infos[n]['cache_bytes']==gold[index]['arms'][n]['cache_bytes']
                    assert infos[n]['late_refresh']==gold[index]['arms'][n]['refresh']
                    # Tensor checks deliberately outside timing, including membrane.
                    exact_tree(predictions[n],predictions[n]);exact_tree(states[n],states[n])
                    if repeat==0:
                        set_statistics(models[n], True)
                        ref,ref_states[n],ref_info=models[n].step(ego,ref_states[n],meta)
                        set_statistics(models[n], False)
                        exact_tree(ref,predictions[n]);exact_tree(ref_states[n],states[n]);compare_info(ref_info,infos[n])
                log.write(json.dumps(dict(round=repeat,frame=meta['frame'],supervised=bool(meta['supervised']),order=order,
                    warmup=index<16,ms=elapsed,core_conv_flops=cost,cache_bytes={n:infos[n]['cache_bytes'] for n in names},
                    refresh=infos['strict']['late_refresh'],reference_exact=repeat==0))+'\n')
                if (index+1)%64==0:
                    if a.suite == 'timing':occupancy.append(check_exclusive(a.gpu));write_json(out/'occupancy.json',occupancy)
                    write_json(out/'status.json',dict(status='RUNNING',phase=a.suite,round=repeat,scans=index+1,total=frames))
            assert index+1==frames
            if a.suite == 'timing':
                rounds.append({n:float(np.mean(local[n])) for n in names})
                occupancy.append(check_exclusive(a.gpu));write_json(out/'occupancy.json',occupancy)
    assert all(model_digest(m)==before[n] for n,m in models.items())
    assert all(digest(pilot/(n+'_target.pth'))==sha for n,sha in shas.items())
    metrics={}
    for n,m in models.items():
        old=prior['evaluation']['metrics'][n]
        metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),
            core_conv_flops_mean=float(np.mean(costs[n])) if costs[n] else old['core_conv_flops_mean'],
            model_step_ms_mean=float(np.mean(timings[n])) if timings[n] else None,
            ap50=old['ap50'] if a.suite=='timing' else None,ap70=old['ap70'] if a.suite=='timing' else None,
            cache_bytes=old['cache_bytes'],scale_buffer_bytes=0 if n=='strict' else 1024,
            samples=len(timings[n]),ap_source='R109, inherited after full-stream parity' if a.suite=='timing' else 'Not measured; prefix parity does not establish full AP')
    write_json(out/'report.json',dict(status='COMPLETED',suite=a.suite,source_report_sha256=contract['source_report_sha256'],
        checkpoint_sha256=prior['checkpoint_sha256'],exact_frames=frames,metrics=metrics,rounds=rounds,
        all_weights_bn_scale_unchanged=True,timing_valid=a.suite=='timing',trained=False,ap_recomputed=False,
        boundary='Same GPU paired model timing only when exclusive. Dense Conv simulation, not sparse hardware. Occupancy sampled, not continuous.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','pilot','full-reference','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--suite',choices=('gate','timing'),required=True);p.add_argument('--gate-report')
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.suite=='timing' and not a.gate_report:p.error('Timing requires completed --gate-report')
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Runtime validation budget')))
        signal.alarm(1800 if a.suite=='gate' else 5400)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    if a.suite=='timing':check_exclusive(a.gpu)
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<6000:raise RuntimeError('Requires at least6GB free; no interference with resident jobs')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.benchmark_int_spatial_spike']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=os.environ.get('PYTHONPATH',str(Path.cwd())),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
