"""Fixed first R-297 clip, deep carry and signed projection equivalence."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.point_pillar_int import PointPillarINT
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_factorized_readout import compact_readout
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.tools.check_int_deep_block_adapt_transfer import average_precision
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.eval_int_deep_block_membrane import load_quantum, score_arm
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


def read(path):
    return json.loads(path.read_text())


def build_source(inputs, cfg):
    for key, filename in [('checkpoint', 'parent.pth'), ('trained', 'protected_light_final.pth'),
                          ('readout', 'readout.pth')]:
        assert digest(inputs / filename) == SOURCE_SHA[key]
    teacher = PointPillarINT(cfg['model']['args'], 'gru').eval()
    base = PyramidStateDetector(teacher, cfg).eval()
    base.load_state_dict(torch.load(inputs / 'parent.pth', map_location='cpu')['model'], strict=True)
    source = ProtectedMultiscaleRefreshDetector(base, True, period=5, max_age=.55)
    load_targets(source, inputs / 'protected_light_final.pth', SOURCE_SHA['trained'])
    source.core.readout, _ = compact_readout(base.core.readout, 128)
    source.core.readout.load_state_dict(torch.load(inputs / 'readout.pth', map_location='cpu')['readout'])
    return source.eval().requires_grad_(False).cuda()


@torch.no_grad()
def run(inputs, out):
    torch.manual_seed(303)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ref = read(inputs / 'r297.json')
    for name, key in [('plan.json','plan'), ('traces.json','traces'), ('quantum.json','calibration')]:
        assert digest(inputs / name) == ref['inputs_sha256'][key]
    rows = read(inputs / 'rows.json')
    plan = read(inputs / 'plan.json')
    traces = read(inputs / 'traces.json')['val'][0]
    assert [r['frame'] for r in rows] == plan['clips'][0] == [t['frame'] for t in traces]
    members = {}
    for r in rows:
        for key in ('pcd', 'label'):
            if r[key]:
                p = inputs / key / Path(r[key]).name
                members[str(p.relative_to(inputs))] = digest(p)
                r[key] = str(p)
    cfg = load_yaml(str(inputs / 'config.yaml'))
    cfg['data_augment'] = []
    source = build_source(inputs, cfg)
    source_hash = model_digest(source)
    assert source_hash == ref['source_frozen_digest'], source_hash
    q, _, _ = load_quantum(inputs / 'quantum.json', plan, 512)
    projectors = {mode: DeepSpikeProjection(source.core.deblocks[2], q, mode)
                  for mode in ('dense','amplitude','unit')}
    models = {mode: DeepBlockCache(source, 8, 'light', crop_first=True,
                                   planner=plan_host_coalesced_refresh, deep_projector=projectors.get(mode))
              for mode in ('continuous', *projectors)}
    write_json(out / 'contract.json', dict(source=source_hash, weights=SOURCE_SHA,
               code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               inputs={p.name:digest(p) for p in inputs.iterdir() if p.is_file()},
               members=members, device=torch.cuda.get_device_name(), torch=torch.__version__,
               exact_fields=['deep','deep_projection','psm','rm','dm'], mode='carry',
               refresh='full dense, exact double reference', params=sum(p.numel() for p in source.parameters())))
    data, stream = loader(rows, cfg, 0, 303)
    states = dict.fromkeys(models)
    stats = {m:{t:dict(tp=[], fp=[], score=[], gt=0) for t in (.5,.7)} for m in models}
    with (out/'frames.jsonl').open('x', buffering=1) as raw:
        for sample, trace in zip(stream, traces):
            meta = sample['meta']
            assert meta['frame'] == trace['frame'] and meta['supervised'] == trace['supervised']
            ego = to_device(sample['ego'], torch.device('cuda'))
            spatial = spatial_input(source, ego)
            masks = trace_masks(trace, spatial.device)
            gt = data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            preds, costs = {}, {}
            for name, model in models.items():
                (preds[name], states[name], info), costs[name] = measured_step(
                    model, spatial, states[name], meta, local_request(masks))
                assert trace['refresh'] == info['full_refresh']
                if name in projectors:
                    for key in ('deep_projection','value','coarse','deep','deep_age'):
                        assert torch.equal(states[name][key], states['dense'][key]), (name,key,meta['frame'])
                    for key in preds[name]:
                        assert torch.equal(preds[name][key], preds['dense'][key]), (name,key,meta['frame'])
                    assert torch.equal(states[name]['deep'], states['continuous']['deep'])
                if gt is not None:
                    score_arm(data.post, dict(ego=ego), preds[name], gt, stats[name])
            ledger = projectors['unit'].ledger
            assert ledger == projectors['amplitude'].ledger == projectors['dense'].ledger
            assert costs['continuous'] - costs['unit'] == 2*ledger['dense_mac']
            record = dict(frame=meta['frame'], refresh=trace['refresh'], supervised=meta['supervised'],
                          core_float_conv_flops=costs, cache_bytes=states['unit']['deep_projection'].numel()*8,
                          phase_write_cells=int(masks['written'].sum()) if masks is not None else states['unit']['deep'].shape[-2]*states['unit']['deep'].shape[-1],
                          **ledger)
            raw.write(json.dumps(record)+'\n')
            print(meta['frame'], ledger['units'], flush=True)
    assert sum(1 for _ in (out/'frames.jsonl').open()) == len(rows)
    metrics = {m:{t:average_precision(np.asarray(s['score']), np.asarray(s['tp']), s['gt'])
                  for t,s in arm.items()} for m,arm in stats.items()}
    write_json(out/'ap_records.json',stats)
    write_json(out/'report.json',dict(scans=len(rows), labelled=sum(r['supervised'] for r in rows),
                                      gt=stats['dense'][.7]['gt'], metrics=metrics, status='COMPLETED'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.inputs, args.output)
