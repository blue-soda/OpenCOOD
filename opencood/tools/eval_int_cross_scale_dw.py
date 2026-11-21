"""Replay the frozen local-hold detector with a directly updated cross-scale receiver."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_cross_scale_dw import CrossScaleCache
from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate, conv_costs
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


@torch.no_grad()
def run(args):
    inputs, prior, out = args.inputs, args.prior, args.output
    source, cfg, rows, traces, starts, members, q = prepare_stream(inputs, args.scans)
    assert members == read(prior/'contract.json')['members']
    checks = [json.loads(line) for line in (prior/'frames.jsonl').read_text().splitlines()]
    per = conv_costs(source.core)
    channel = source.core.deep_light[0].in_channels
    modes = ('dense', 'amplitude') if args.quality else ('dense', 'unit')
    if args.sparse_cold:
        modes = ('dense','static','sparse_amplitude','sparse_unit')
    models = {}
    for mode in modes:
        model = CrossScaleCache(source, mode, planner=plan_host_coalesced_refresh,
                                deep_projector=DeepSpikeProjection(source.core.deblocks[2], q, 'amplitude'))
        model.coarse_executor = FineInnovationUpdate(False)
        model.coarse_interface = model.coarse_projector = MiddleDecayProjection(source.core.deblocks[1], 'zero', 'amplitude')
        models[mode] = model
    data, stream = loader(rows, cfg, 0, 303)
    write_json(out/'contract.json', dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               reference=digest(prior/'frames.jsonl'), trace=digest(inputs/'traces.json'), modes=modes,
               params=sum(p.numel() for p in source.parameters()), channels=channel, pointwise_out=source.core.deep_light[2].out_channels,
               quality=args.quality, quantization='per channel int16; middle q=.25', gpu=torch.cuda.get_device_name()))
    n = 0
    with (out/'frames.jsonl').open('x', buffering=1) as log:
        for sample, trace, check in zip(stream, traces, checks):
            meta = sample['meta']
            frame, reason = meta['frame'], trace['refresh']
            assert frame == trace['frame'] == check['frame']
            if frame in starts:
                states = dict.fromkeys(modes)
            ego = to_device(sample['ego'], torch.device('cuda'))
            spatial = spatial_input(source, ego)
            gt = data.post.generate_gt_bbx(dict(ego=ego)) if args.quality and meta['supervised'] else None
            request = local_request(trace_masks(trace, spatial.device))
            row = dict(frame=frame, scene=meta['scene'], refresh=reason, arms={})
            baseline = check['arms']['zero_zero']
            for mode, model in models.items():
                prev = states[mode]
                assert model.refresh_reason(prev, meta) == reason
                model.coarse_projector.begin(prev, reason)
                (pred, state, info), other = measured_step(model, spatial, prev, meta, request)
                states[mode] = state
                edge = info.get('edge', {})
                cost = 0
                if reason is None:
                    executor = model.coarse_executor
                    cost = sum(per[k]*v for k,v in executor.counts.items())
                    assert executor.counts == baseline['support']
                    executor.inputs = None
                    cache, valid = state['edge_cache']
                    row['cache_bytes'] = cache.numel()*cache.element_size()+valid.numel()
                    row['valid_sites'] = int(valid.sum())
                    assert edge.get('sites',0) == round(info['stage_conv_ratio']*valid.numel())
                assert other + cost + 2*9*channel*edge.get('sites',0) == baseline['core'], frame
                entry = dict(edge=edge, core_float_conv=other+cost, baseline_core=baseline['core'])
                if mode == 'dense':
                    reference = pred
                    if gt is not None:
                        row.update(score(data.post, ego, pred, gt))
                else:
                    assert set(reference) == set(pred)
                    assert all(torch.equal(reference[k], pred[k]) for k in pred), frame
                    for key in ('deep','coarse','coarse_projection','deep_projection'):
                        assert torch.equal(states['dense'][key],state[key]), (frame,key)
                    if reason is None:
                        assert all(torch.equal(x,y) for x,y in zip(states['dense']['edge_cache'],state['edge_cache'])), frame
                row['arms'][mode] = entry
            log.write(json.dumps(row)+'\n')
            n += 1
    assert n == len(rows) == len(checks) == 272
    write_json(out/'report.json', dict(status='COMPLETED', scans=n, modes=modes, state_and_head_equal=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('inputs','scans','prior','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--quality', action='store_true')
    parser.add_argument('--sparse-cold', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    run(args)

