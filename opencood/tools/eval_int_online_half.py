"""Evaluate three causal task-state detectors with separate live residual selectors."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate, conv_costs
from opencood.models.sub_modules.int_cross_scale_dw import CrossScaleCache
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_fine_codebook import FineCodebook
from opencood.models.sub_modules.int_fine_float_receiver import FloatQuotaReceiver
from opencood.models.sub_modules.int_fine_registration import ReceiverGuide
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionCache
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_codebook_fraction import FullAmplitude
from opencood.tools.eval_int_deep_block_online import CachedResidualPolicy, ObservedPolicy
from opencood.tools.eval_int_fusion_cache import complete_scenes
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import compact_group_bucket, spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json


def build(source, q, edge):
    def refresh(value):
        return edge(value) if edge.phase == 'refresh' else value

    def transmit(value, active):
        edge.active = active
        return edge(value)

    core = source.core
    model = CrossScaleCache(source, 'sparse_amplitude', planner=plan_host_coalesced_refresh,
                            deep_projector=DeepSpikeProjection(core.deblocks[2], q, 'amplitude'),
                            fine_to_coarse=refresh if edge else None)
    model.coarse_interface = model.coarse_projector = MiddleDecayProjection(core.deblocks[1], 'zero', 'amplitude')
    model.coarse_executor = FineInnovationUpdate(False, transmit if edge else None)
    return model


def indices(mask):
    return mask.flatten().nonzero().flatten().tolist()


@torch.no_grad()
def run(args):
    source, cfg, short, _, _, _, q = prepare_stream(args.inputs, args.scans)
    rows, traces, hashes = complete_scenes(args, short, remainder=args.remainder)
    first_segment = read(args.long_base/'r309_full_plan.json')['ordered_stream'][0]['segment']
    book = read(args.book)
    assert book['levels'] == [3, 6, 13, 28] and book['cap'] == 428149
    cap = (book['cap'] // 2)*args.eighths//8
    edges = dict(carry=FineCodebook(source.bounds, cap, book['levels']),
                 amplitude=FullAmplitude(source.bounds, 'amplitude', cap), continuous=None)
    core = source.core
    guide = None
    if args.registration:
        edges = dict(carry=edges['carry'], amplitude=edges['amplitude'],
                     float_guided=FloatQuotaReceiver(source.bounds,cap),float_ego=FloatQuotaReceiver(source.bounds,cap))
        guide = ReceiverGuide()
        for name,edge in edges.items():
            if name!='float_ego':
                edge.transport=guide.propagate
    fusion = core.readout.layers[0].double_conv[0]
    layer = NonzeroFusionCache(fusion[0]).eval()
    fusion[0] = layer
    models = {k: build(source, q, edge) for k, edge in edges.items()}
    per = conv_costs(core)
    data, stream = loader(rows, cfg, 0, 303)
    factory = partial(CachedResidualPolicy, compiler=compact_group_bucket)
    write_json(args.output/'contract.json', dict(native=hashes, code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        book=digest(args.book), cap=cap, params=sum(p.numel() for p in source.parameters()), optimizer_steps=0,
        remainder=args.remainder, registration=args.registration, modes=list(edges), eighths=args.eighths,
        scenes=sorted({r['scene'] for r in rows}), selector='per-arm live CachedResidualPolicy / compact_group_bucket'))
    with (args.output/'frames.jsonl').open('x', buffering=1) as log:
        for i, sample in enumerate(stream):
            meta = sample['meta']
            trace = traces[i]
            if i == 0 or meta['segment'] != rows[i-1]['segment']:
                # Preserve the source protocol's scene-reset vs first-scan semantics.
                initial = None if rows[i]['segment'] == first_segment else dict(scene='')
                states = dict.fromkeys(models, initial)
                policies = {k: ObservedPolicy(model, factory) for k, model in models.items()}
            ego = to_device(sample['ego'], torch.device('cuda'))
            spatial = spatial_input(source, ego)
            gt = data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            row = dict(frame=meta['frame'], scene=meta['scene'], segment=meta['segment'], arms={})
            assert row['frame'] == trace['frame']
            for name, model in models.items():
                prev, edge, policy = states[name], edges[name], policies[name]
                reason = model.refresh_reason(prev, meta)
                if edge is not None:
                    edge.begin(prev, meta, spatial.shape[-2:], reason)
                model.coarse_projector.begin(prev, reason)
                layer.begin(prev, reason, True)
                policy.observation = None
                use_guide = guide is not None and name!='float_ego'
                if use_guide:
                    hook = core.observation.register_forward_hook(guide.observe)
                (pred, state, info), other = measured_step(model, spatial, prev, meta, policy)
                if use_guide:
                    hook.remove()
                    guide.save(state)
                assert info['full_refresh'] == reason == trace['refresh']
                stats = dict(refresh=reason)
                if reason is None:
                    valid, mandatory, optional = policy.observation
                    written = state['deep_age'] == 0
                    assert torch.equal(written, mandatory if optional is None else mandatory | optional)
                    assert not bool((~valid & ~written).any())
                    stats.update(mandatory=indices(mandatory), written=indices(written), selection=policy.policy.record,
                                 invalid=int((~valid).sum()), historical=trace['written_indices'])
                    exe = model.coarse_executor
                    other += sum(per[k]*v for k,v in exe.counts.items())
                    stats['middle_mask_sha256'] = hashlib.sha256(exe.skip.cpu().numpy().tobytes()).hexdigest()
                    exe.inputs = None
                    if edge is not None:
                        stats.update({k:edge.ledger[k] for k in ('addresses','bytes','units','sent_abs_float','demand_addresses') if k in edge.ledger})
                        if use_guide:
                            stats['registration']=dict(guide.cost)
                        assert stats['addresses'] == min(cap,edge.ledger['demand_addresses'])
                else:
                    assert policy.observation is None
                if edge is not None:
                    edge.commit(state,False)
                state['fusion_cache'] = layer.new
                stats['core'] = other + layer.ledger['float_conv']
                if gt is not None:
                    stats.update(score(data.post, ego, pred, gt, record_matches=True))
                row['arms'][name] = stats
                states[name] = state
            log.write(json.dumps(row)+'\n')
    assert i+1 == len(rows)
    write_json(args.output/'report.json',dict(status='COMPLETED', scans=len(rows), labels=sum(r['supervised'] for r in rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('inputs','scans','book','vehicle','long-base','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--remainder', action='store_true')
    parser.add_argument('--registration',action='store_true')
    parser.add_argument('--eighths',type=int,choices=(7,8),default=8)
    args = parser.parse_args()
    if (args.remainder and args.registration) or (args.eighths!=8 and not args.registration):
        parser.error('registration budgets are restricted to the development comparison')
    args.output.mkdir(parents=True, exist_ok=False)
    run(args)

