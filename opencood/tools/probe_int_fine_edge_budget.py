"""Training-only shadow ledger for a fine sender and the next pointwise receiver."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate, supports
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.tools.calibrate_int_middle_budget import detector, training_rows
from opencood.tools.check_int_deep_spike_stream import build_source, read
from opencood.tools.eval_int_deep_block_membrane import load_quantum
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


@torch.no_grad()
def run(inputs, training, out):
    torch.manual_seed(303)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    plan = read(inputs/'plan.json')
    rows,traces,starts = training_rows(inputs,training,plan)
    cfg = load_yaml(str(inputs/'config.yaml'))
    cfg['data_augment'] = []
    source = build_source(inputs,cfg)
    q,_,_ = load_quantum(inputs/'quantum.json',plan,512)
    model = detector(source,q,'carry',.25)
    model.coarse_executor = FineInnovationUpdate(False)
    model.coarse_interface = model.coarse_projector = MiddleDecayProjection(source.core.deblocks[1],'zero','amplitude')
    spatial_conv,relu,projection = source.core.coarse_light
    contract = dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                    train=digest(training/'train_plan.json'),trace=digest(inputs/'traces.json'),
                    gpu=torch.cuda.get_device_name(),params=sum(x.numel() for x in source.parameters()),
                    channels=projection.in_channels, fanout=projection.out_channels,quantum=.25,labels_used=0)
    write_json(out/'contract.json',contract)
    _,stream = loader(rows,cfg,0,303)
    n = 0
    with (out/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace in zip(stream,traces):
            meta = sample['meta']
            frame,reason = meta['frame'],trace['refresh']
            assert not meta['supervised'] and frame==trace['frame']
            if frame in starts:
                state = None
            prev = state
            assert model.refresh_reason(prev,meta)==reason
            model.coarse_projector.begin(prev,reason)
            ego = to_device(sample['ego'],torch.device('cuda'))
            spatial = spatial_input(source,ego)
            _,state,_ = model.step_spatial(spatial,prev,meta,local_request(trace_masks(trace,spatial.device)))
            fine = state['value']
            levels = (fine/.25).round()
            row = dict(frame=frame,scene=meta['scene'],refresh=reason,
                       fine_cells=levels.numel(),fine_nonzero=int(levels.count_nonzero()))
            if reason is not None:
                last_projection_input = None
            else:
                support = supports(~model.coarse_executor.skip)[2]
                assert int(support.sum())==model.coarse_executor.counts['radius2']
                model.coarse_executor.inputs = None
                args = (prev['pose'],meta['pose'])
                area = block_lidar_range(source.bounds,spatial.shape[-2:],fine.shape[-2:],2)
                old = warp_bev(last_levels,*args,area)[0]
                delta = levels-old
                # Shadow activations feed only this accounting path.
                x = relu(spatial_conv(levels*.25))
                current = (x!=0) & support
                changes = current
                valid = torch.zeros_like(support)
                if last_projection_input is not None:
                    area = block_lidar_range(source.bounds,spatial.shape[-2:],support.shape[-2:],4)
                    aligned = warp_bev(last_projection_input,*args,area)[0]
                    valid = (aligned[:,-1:]!=0)&support
                    changes = torch.where(valid,x!=aligned[:,:-1],current)
                row.update(fine_events=int(delta.count_nonzero()),fine_units=int(delta.abs().sum(dtype=torch.float64)),
                           sites=int(support.sum()),mapped_sites=int(valid.sum()),static_nonzero=int(current.sum()),
                           changed_or_cold=int(changes.sum()),cold_nonzero=int((current&~valid).sum()))
                last_projection_input = torch.cat((x,support.float()),dim=1)
            last_levels = levels
            log.write(json.dumps(row)+'\n')
            n += 1
    assert n==len(rows)==256
    write_json(out/'report.json',dict(status='COMPLETED',scans=n,labels_used=0,shadow_only=True))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    for name in ('inputs','training','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    run(a.inputs,a.training,a.output)

