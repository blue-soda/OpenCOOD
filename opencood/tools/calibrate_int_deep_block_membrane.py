"""Calibrate a deep task-state quantum from label-free training-stream activations."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import TILE, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, add_tiles, build, masks_by_tile, spatial_input
from opencood.tools.run_int_ego import to_device, write_json


MIN_EXPONENT = -16
RMS_FRACTION = .5
RULE = 'Each channel: nearest power of two to 0.5 * RMS, exponent floored at -16; no label, AP or validation-based choice.'


def run(args, out):
    plan = checked_plan(args.plan)
    assert plan['split'] == 'train'
    args.kind = 'quality'
    source, cfg, rows, provenance = build(args)
    assert provenance['split'] == 'train'
    source.period, source.max_age = 5, .55
    frozen = model_digest(source)
    cache = DeepBlockCache(source, TILE, 'light')
    identity = dict(split='train', source_sha256=SOURCE_SHA,
                    plan_sha256=digest(Path(args.plan)), scenes=plan['scenes'],
                    period=source.period, max_age=source.max_age, tile=TILE,
                    code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip())
    write_json(out/'contract.json', dict(**identity, rule=RULE, torch=torch.__version__,
        device=torch.cuda.get_device_name(), scope='Label-free geometry trace with independently recurrent ANN state. All deep readout lattice elements, including retained cells.'))
    scans, elements, core_flops = 0, 0, 0
    total_squares = np.zeros(source.core.deep_light[-1].out_channels, dtype=np.float64)
    with torch.no_grad(), (out/'frames.jsonl').open('x', buffering=1) as raw:
        for clip, ids, _, stream in clip_streams(plan, rows, cfg):
            state = None
            for index, sample in enumerate(stream):
                meta = sample['meta']
                assert meta['frame'] == ids[index]
                spatial = spatial_input(source, to_device(sample['ego'], torch.device('cuda')))
                request = None
                if cache.refresh_reason(state, meta) is None:
                    frame = cache.prepare(spatial, state, meta)
                    required = cache.mandatory(frame).cpu()
                    keys = list(masks_by_tile(required, TILE))
                    if keys:
                        request = add_tiles(torch.zeros_like(required), (keys[index*7 % len(keys)],), TILE).cuda()
                (_, state, info), flops = measured_step(cache, spatial, state, meta, request)
                core_flops += flops
                deep = state['deep']
                per_channel = deep.square().sum(dim=(0,2,3)).cpu().numpy()
                assert np.isfinite(per_channel).all()
                count = deep.shape[0] * deep.shape[-2] * deep.shape[-1]
                elements += count
                total_squares += per_channel
                scans += 1
                raw.write(json.dumps(dict(frame=ids[index], clip=clip, elements_per_channel=count,
                    refresh=info['full_refresh'], core_conv_flops=flops,
                    square_sums=per_channel.tolist()), allow_nan=False)+'\n')
            write_json(out/'status.json', dict(status='RUNNING', scans=scans, clips=clip+1))
    assert model_digest(source) == frozen and scans == sum(map(len,plan['clips']))
    rms = np.sqrt(total_squares / elements)
    exponent = np.rint(np.log2(np.maximum(rms * RMS_FRACTION, 2.**MIN_EXPONENT)))
    quantum = np.exp2(exponent)
    assert np.isfinite(quantum).all() and (quantum >= 2.**MIN_EXPONENT).all()
    write_json(out/'quantum.json', dict(**identity, rule=RULE, scans=scans,
        elements_per_channel=elements, rms=rms.tolist(), quantum=quantum.tolist()))
    write_json(out/'report.json', dict(status='COMPLETED', scans=scans, channels=len(quantum),
        quantum_min=float(quantum.min()), quantum_median=float(np.median(quantum)),
        quantum_max=float(quantum.max()), channels_below_historical_quantum=int((quantum<.25).sum()),
        elements_per_channel=elements, parameters=sum(p.numel() for p in source.parameters()),
        core_conv_flops_mean=core_flops/scans, model_step_ms_mean=None, ap50=None, ap70=None,
        weights_unchanged=True))
    write_json(out/'status.json', dict(status='COMPLETED', scans=scans))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('root','checkpoint','trained','readout','plan','output'):
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'status.json', dict(status='RUNNING', phase='inputs'))
    try:
        run(args, out)
    except Exception as exc:
        write_json(out/'status.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
