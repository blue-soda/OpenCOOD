"""Same-card deep projection timing on previously frozen residual-mask clips."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess
import time

import numpy as np
import torch

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import TILE, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.time_int_deep_block_cache import PASSES, fingerprint
from opencood.tools.time_int_periodic_compact import exclusive
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


MODES = ('p5', 'crop', 'projection', 'phase_index')
WARMUP = 8
DEVICE = torch.device('cuda')


def inputs(args):
    folder = Path(args.final_run)
    paths = dict(plan=Path(args.plan), contract=folder/'contract.json',
                 traces=folder/'fixed_traces.json', frames=folder/'eval_08'/'frames.jsonl')
    hashes = {k:digest(path) for k,path in paths.items()}
    prior = json.loads(paths['contract'].read_text())['identity']
    assert prior['plans']['val']==hashes['plan'] and prior['trace_sha256']==hashes['traces']
    assert prior['source_sha256']==SOURCE_SHA and (prior['period'],prior['max_age'],prior['tile'])==(5,.55,TILE)
    plan = checked_plan(args.plan)
    assert plan['split']=='val' and len(plan['clips'])==6 and sum(map(len,plan['clips']))==272
    traces = json.loads(paths['traces'].read_text())['val']
    assert [[r['frame'] for r in clip] for clip in traces]==plan['clips']
    costs = {(r['clip'],r['frame']):r['core_conv_flops']
             for r in map(json.loads,paths['frames'].read_text().splitlines())}
    assert set(costs)=={(i,f) for i,clip in enumerate(plan['clips']) for f in clip}
    return paths,hashes,plan,traces,costs


def clips(plan, rows, cfg, traces, uuid):
    for clip, ids, _, data in clip_streams(plan,rows,cfg):
        def frames():
            for i,sample in enumerate(data):
                if i%16==0:
                    exclusive(uuid)
                reference = traces[clip][i]
                assert sample['meta']['frame']==ids[i]==reference['frame']
                yield i,sample,reference,trace_masks(reference,DEVICE)
        yield clip,frames()


def audit(source, models, cfg, rows, plan, traces, costs, uuid, out):
    conv = source.core.deblocks[2][0]
    flop_per_site = 2*conv.in_channels*conv.out_channels*conv.kernel_size[0]*conv.kernel_size[1]
    proofs = {}
    with (out/'equivalence.jsonl').open('x',buffering=1) as raw:
        for clip, data in clips(plan,rows,cfg,traces,uuid):
            states = dict.fromkeys(MODES)
            for i,sample,ref,masks in data:
                meta = sample['meta']
                frame = meta['frame']
                spatial = spatial_input(source,to_device(sample['ego'],DEVICE))
                prior_cost = costs[clip,frame]
                arms, preds = {}, {}
                for name,model in models.items():
                    request = None if name=='p5' else local_request(masks)
                    (preds[name],states[name],info),flops = measured_step(model,spatial,states[name],meta,request)
                    assert info['full_refresh']==ref['refresh']
                    arms[name] = dict(core_conv_flops=flops,
                                      fingerprint=fingerprint(preds[name],states[name]) if i%16==0 else None)
                base = states['crop']
                written = int((base['deep_age']==0).sum())
                slots = base['deep_age'].numel()
                assert arms['p5']['core_conv_flops']==prior_cost['p5']
                assert arms['crop']['core_conv_flops']==prior_cost['frozen_local']
                for name in ('projection','phase_index'):
                    assert arms[name]['core_conv_flops']==prior_cost['frozen_local']-(slots-written)*flop_per_site
                    assert all(torch.equal(states[name][k],base[k]) for k in (*source.state_keys,'deep_age'))
                    assert all(torch.equal(preds[name][k],preds['crop'][k]) for k in preds['crop'])
                assert torch.equal(states['projection']['deep_projection'],states['phase_index']['deep_projection'])
                proof = dict(clip=clip,frame=frame,refresh=ref['refresh'],written=written,slots=slots,
                             invalid=0 if masks is None else int((~masks['valid']).sum()),arms=arms)
                proofs[clip,frame] = proof
                raw.write(json.dumps(proof,allow_nan=False)+'\n')
            write_json(out/'status.json',dict(status='RUNNING',phase='audit',clips=clip+1))
    return proofs


def time_clips(source, models, cfg, rows, plan, traces, proofs, uuid, out):
    result = []
    with (out/'frames.jsonl').open('x',buffering=1) as raw:
        for repeat in range(PASSES):
            exclusive(uuid)
            for clip,data in clips(plan,rows,cfg,traces,uuid):
                states = dict.fromkeys(MODES)
                for i,sample,ref,masks in data:
                    meta = sample['meta']
                    ego = to_device(sample['ego'],DEVICE)
                    shift = (i+clip+repeat)%len(MODES)
                    order = MODES[shift:]+MODES[:shift]
                    gold = proofs[clip,meta['frame']]
                    arms = {}
                    for name in order:
                        request = None if name=='p5' or masks is None else masks['optional']
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        spatial = spatial_input(source,ego)
                        pred,states[name],info = models[name].step_spatial(spatial,states[name],meta,requests=request)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter()-start)*1000.
                        assert info['full_refresh']==ref['refresh']
                        identity = gold['arms'][name]
                        if identity['fingerprint'] is not None:
                            assert fingerprint(pred,states[name])==identity['fingerprint']
                        arms[name] = dict(model_step_ms=elapsed,core_conv_flops=identity['core_conv_flops'])
                    record = dict(repeat=repeat,clip=clip,index=i,frame=meta['frame'],order=order,
                                  included=i>=WARMUP,refresh=ref['refresh'],arms=arms)
                    result.append(record)
                    raw.write(json.dumps(record,allow_nan=False)+'\n')
                write_json(out/'status.json',dict(status='RUNNING',phase='timing',repeat=repeat,clips=clip+1))
    exclusive(uuid)
    return result


def summarize(records):
    def stats(mode, data):
        samples = [r['arms'][mode]['model_step_ms'] for r in data]
        return dict(samples=len(samples),mean_ms=mean(samples),p95_ms=float(np.percentile(samples,95)),
                    mean_core_conv_flops=mean(r['arms'][mode]['core_conv_flops'] for r in data))
    active = [r for r in records if r['included']]
    ordinary = [r for r in active if r['refresh'] is None]
    return {name:dict(overall=stats(name,active),ordinary=stats(name,ordinary),
                     pass_means=[stats(name,[r for r in active if r['repeat']==j])['mean_ms'] for j in range(PASSES)],
                     clip_means=[stats(name,[r for r in active if r['clip']==j])['mean_ms'] for j in range(6)])
            for name in MODES}


@torch.no_grad()
def run(args,out):
    exclusive(args.uuid)
    paths,hashes,plan,traces,costs = inputs(args)
    args.kind = 'geometry'
    source,cfg,rows,_ = build(args)
    source.period,source.max_age = 5,.55
    weights = model_digest(source)
    models = {'p5':source}
    for name in MODES[1:]:
        projector = None if name=='crop' else DeepPhaseProjection(source.core.deblocks[2], indexed=name=='phase_index')
        models[name] = DeepBlockCache(source,TILE,'light',planner=plan_host_coalesced_refresh,crop_first=True,
                                      deep_projector=projector)
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        device=torch.cuda.get_device_name(),uuid=args.uuid,torch=torch.__version__,source_sha256=SOURCE_SHA,
        inputs_sha256=hashes,passes=PASSES,warmup_per_clip=WARMUP,arms=MODES,
        scope='Pillar to heads; fixed residual optional masks, no loader, H2D, postprocessing, or mask scoring. AP and energy unmeasured.'))
    proofs = audit(source,models,cfg,rows,plan,traces,costs,args.uuid,out)
    assert len(proofs)==272
    records = time_clips(source,models,cfg,rows,plan,traces,proofs,args.uuid,out)
    assert len(records)==272*PASSES and model_digest(source)==weights
    assert all(digest(paths[k])==h for k,h in hashes.items())
    write_json(out/'report.json',dict(status='COMPLETED',timing_valid=True,exact_scans=len(proofs),
        parameters=sum(p.numel() for p in source.parameters()),metrics=summarize(records),
        scenes=[s[0] for s in plan['scenes']],weights_unchanged=True))
    write_json(out/'status.json',dict(status='COMPLETED',scans=len(proofs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','plan','final-run','uuid','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='inputs'))
    try:
        run(args,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',timing_valid=False,error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
