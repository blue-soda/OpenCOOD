"""Validate mature-initialized dense spatial bases before temporal reuse.

Rank9 is a function-preservation control, rank6 is a compression diagnostic.
Neither job reports random-mask kernel tests as regional detector accuracy.
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
import torch
from opencood.models.sub_modules.int_compact_readout import compact_shrink
from opencood.models.sub_modules.int_projected_gru import projected_gru
from opencood.models.sub_modules.int_basis_spatial_state import SpatialBasisStage
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout
from opencood.models.sub_modules.int_light_repair import LightRepair
from opencood.tools.evaluate_int_compact_repair import completed_best
from opencood.tools.train_int_projected_gru import make_modes, evaluate
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest


def worker(a):
    out = Path(a.output); write_json(out/'status.json', dict(status='RUNNING', phase='setup'))
    for k, v in dict(mode='gru', stage='pilot', device='cuda', content_calibration=None, periodic_last=0, max_age=.25).items():
        setattr(a, k, v)
    base, cfg, val, source = setup(a)
    rb, readout = completed_best(a.readout_run)
    artifacts = []
    for root, expected in ((a.memory_run, '1a267575ef735f203526fa3b8196334d32d9ff8a9080449c3b6206247913e5ef'),
                           (a.repair_run, '9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd')):
        root = Path(root)
        assert json.loads((root/'status.json').read_text())['status'] == 'COMPLETED'
        best = json.loads((root/'best.json').read_text())
        assert digest(best['checkpoint']) == best['sha256'] == expected
        artifacts.append((best, torch.load(best['checkpoint'], map_location='cpu')))
    (mb, memory), (pb, repaired) = artifacts
    assert memory['contract']['data'] == repaired['contract']['data'] == source['contract']
    assert memory['contract']['source']['sha256'] == readout['contract']['source_sha256'] == source['checkpoint']['sha256']
    assert repaired['contract']['readout'] == rb
    base.shrink_conv = compact_shrink(base.shrink_conv, 96)[0].cuda()
    base.shrink_conv.load_state_dict(readout['readout'], strict=True)
    base.feature_memory = projected_gru(base.feature_memory, 32)[0].cuda()
    base.feature_memory.load_state_dict(memory['memory'], strict=True); base.eval()
    split = SplitCompactReadout(base.shrink_conv, base.backbone.deblocks[0][0].out_channels).cuda().eval()
    repair = LightRepair(96, width=32).cuda().eval(); repair.load_state_dict(repaired['repair'], strict=True)
    torch.manual_seed(303)
    student = copy.deepcopy(base)
    stage = SpatialBasisStage(base.backbone.resnet.layer0, a.rank).cuda().eval()
    student.backbone.resnet.layer0 = stage
    for m in (base, student, split, repair):
        for p in m.parameters(): p.requires_grad_(False)
    contract = dict(source=source['checkpoint'], data=source['contract'], memory=mb, readout=rb, repair=pb,
        rank=a.rank, retained_weight_energy=stage.retained_weight_energy,
        architecture='Mature stem unchanged. Four per-input spatial SVD depthwise bases followed by pointwise channel mixers; original BN/residual/ReLU preserved. Rank9 algebraically full rank, rank6 approximate.',
        state_design='Four aligned whole post-activation maps (conv1 and block output for each of two blocks). Current spatial bases remain dense; selected expensive mixing has no duplicate halo outputs. Not enabled in this dense validation.',
        stage_parameters=sum(p.numel() for p in stage.parameters()),
        source_stage_parameters=sum(p.numel() for p in base.backbone.resnet.layer0.parameters()),
        protocol='No training, augmentation or new delay. Own e11 FM and original deep anchor states. Four-way full/cache comparison, same GPU rotating order. No new spatial cache in reported AP. Whole validation, no source head-calibration checkpoint used.')
    write_json(out/'contract.json', contract); write_json(out/'provenance.json', source)
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'opencood/test', '-p', 'test_int*.py'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout); assert tests.returncode == 0
    arguments = json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train = json.loads(Path(arguments['manifest']).read_text())['train']
    modes = make_modes(base, student, split, repair, cfg); states = {k: None for k in modes}; costs = {}
    capture = {}
    hook = stage.register_forward_pre_hook(lambda m, x: capture.update(temporal=x[0]))
    maximum = selected_error = 0.; sparse_costs = []; cached_map_bytes = 0
    _, stream = loader(train[:32], cfg, 0, 303)
    try:
        with torch.no_grad():
            for index, sample in enumerate(stream):
                ego = to_device(sample['ego'], torch.device('cuda')); preds = {}
                for name, mode in modes.items():
                    root = base if name.startswith('base') else student
                    counter = ConvCounter(root, torch.nn.ModuleList([split, repair]))
                    try: pred, states[name], info = mode.step(ego, states[name], sample['meta'])
                    finally: counter.close()
                    assert all(torch.isfinite(v).all() for v in pred.values())
                    key = name+('_partial' if info['local_partial'] else '_full')
                    if key in costs: assert costs[key] == counter.total
                    costs[key] = counter.total; preds[name] = pred
                for name in ('full', 'cached'):
                    for key in preds[name]:
                        maximum = max(maximum, float((preds[name][key]-preds['base_'+name][key]).abs().max()))
                        if a.rank == 9:
                            assert torch.allclose(preds[name][key], preds['base_'+name][key], atol=1e-3, rtol=1e-3), maximum
                if index == 0:
                    stem = stage.observe(capture['temporal']); _, maps = stage.dense_from_stem(stem)
                    mask = torch.zeros_like(stem[:, :1], dtype=torch.bool); mask[:, :, ::2, ::2] = True
                    old = [m*.9 for m in maps]
                    counter = ConvCounter(stage, torch.nn.ModuleList())
                    try: selected, _, macs = stage.selected_from_stem(stem, old, mask)
                    finally: counter.close()
                    reference, _, _ = stage.selected_from_stem(stem, old, mask, 'masked_dense')
                    selected_error = float((selected-reference).abs().max())
                    assert torch.allclose(selected, reference, atol=1e-4, rtol=1e-4), selected_error
                    sparse_costs = dict(write_fraction=float(mask.float().mean()),
                        basis_conv_macs=counter.total, selected_mixer_macs=macs,
                        dense_mixer_macs=sum(stem.shape[-2]*stem.shape[-1]*c.mix.weight.numel()
                            for b in stage.blocks for c in (b.conv1, b.conv2)))
                    cached_map_bytes = sum(m.numel()*m.element_size() for m in maps)
    finally: hook.remove()
    write_json(out/'engineering.json', dict(status='PASSED', frames=32, prediction_max_abs_difference=maximum,
        rank9_equivalence_checked=a.rank == 9, selected_vs_masked_dense_max_error=selected_error,
        conv_macs=costs, four_activation_maps_bytes=cached_map_bytes, random_mask_kernel_diagnostic=sparse_costs,
        warning='Kernel diagnostic excludes current stem, cache alignment and selection; not detector sparse FLOPs/AP/latency. Full validation below uses dense stage only.'))
    epoch = out/'epoch_000'; epoch.mkdir()
    checkpoint = epoch/'stage.pth'; torch.save(dict(stage=stage.state_dict(), contract=contract, epoch=0), str(checkpoint))
    scope = 'Dense mature spatial-basis architecture, rank%d, no training or new spatial reuse. Four-mode original deep-cache/full validation. Conv MACsx2; model wall time includes basis/mixing, excludes IO/H2D/postprocessing.' % a.rank
    report = evaluate(base, student, split, repair, cfg, val, epoch, costs, 0,
                      (.7567024378706645, .7555532616222492), scope)
    if a.rank == 9:
        for name in ('full', 'cached'):
            assert abs(report['results'][name]['ap70']-report['results']['base_'+name]['ap70']) < 1e-5, 'Rank9 full-validation AP drift'
    write_json(out/'best.json', dict(epoch=0, checkpoint=str(checkpoint), sha256=digest(checkpoint),
        ap70=report['results']['full']['ap70'], cached_ap70=report['results']['cached']['ap70']))
    write_json(out/'report.json', report); write_json(out/'status.json', dict(status='COMPLETED'))


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'readout-run', 'repair-run', 'memory-run', 'output'): p.add_argument('--'+key, required=True)
    p.add_argument('--rank', type=int, choices=(6, 9), required=True)
    p.add_argument('--gpu', type=int, required=True); p.add_argument('--worker', action='store_true')
    a = p.parse_args(); out = Path(a.output)
    if a.worker:
        def expired(s, f): raise TimeoutError('Spatial basis validation exceeded4 hours')
        signal.signal(signal.SIGALRM, expired); signal.alarm(14400)
        try: worker(a)
        except Exception as exc: write_json(out/'status.json', dict(status='FAILED', error=repr(exc))); raise
        finally: signal.alarm(0)
        return
    free, util = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free) < 22000 or int(util) > 5: raise RuntimeError('GPU occupied')
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    command = [sys.executable, '-m', 'opencood.tools.evaluate_int_basis_spatial']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out/'launch.json', dict(pid=proc.pid, gpu=a.gpu, command=command, created_unix=time.time(), timeout_seconds=14400,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid, output=str(out))))


if __name__ == '__main__': main()
