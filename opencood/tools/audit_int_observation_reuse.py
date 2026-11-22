"""Count fixed-grid reuse opportunities in the frozen fine BEV observation convolutions."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.nn import functional as F

from opencood.tools.train_int_silent_task import (
    digest, loader, prepare_stream, read, spatial_input, to_device, write_json)


def count_taps(mask, layer):
    counts = mask.sum(1, keepdim=True).float()
    kernel = counts.new_ones((1, 1, *layer.kernel_size))
    return F.conv2d(counts, kernel, stride=layer.stride, padding=layer.padding,
                    dilation=layer.dilation).long()


class ObservationAudit:
    def __init__(self):
        self.previous = {}

    def record(self, name, layer, inputs, out):
        x = inputs[0]
        assert layer.groups == 1 and x.shape[0] == 1 and layer.padding_mode == 'zeros'
        nz = count_taps(x != 0, layer)
        assert tuple(nz.shape[2:]) == tuple(out.shape[2:])
        channels = out.shape[1]
        dense = out.numel()*layer.in_channels*layer.kernel_size[0]*layer.kernel_size[1]
        static = int(nz.sum())*channels
        reuse = hybrid = static
        unchanged = 0
        if name in self.previous:
            old, old_out = self.previous[name]
            change = x != old
            diff = count_taps(change, layer)
            clean = diff == 0
            unchanged = int(clean.sum())
            assert bool(((out == old_out) | ~clean).all())
            reuse = int(nz[~clean].sum())*channels
            hybrid = int(torch.minimum(nz,diff).sum())*channels
            if self.verify_windows:
                # Directly enumerate real input taps at a boundary and an interior output.
                for y,z in ((0,0),(out.shape[2]//2,out.shape[3]//2)):
                    actual = delta = 0
                    for ky in range(layer.kernel_size[0]):
                        for kz in range(layer.kernel_size[1]):
                            py = y*layer.stride[0]-layer.padding[0]+ky*layer.dilation[0]
                            pz = z*layer.stride[1]-layer.padding[1]+kz*layer.dilation[1]
                            if 0 <= py < x.shape[2] and 0 <= pz < x.shape[3]:
                                actual += int(x[:,:,py,pz].count_nonzero())
                                delta += int(change[:,:,py,pz].count_nonzero())
                    assert actual == int(nz[0,0,y,z]) and delta == int(diff[0,0,y,z])
        assert 0 <= hybrid <= reuse <= static <= dense
        self.stats[name] = dict(dense_mac=dense, static_mac=static, clean_reuse_mac=reuse,
            ideal_min_mac=hybrid, clean_sites=unchanged, sites=nz.numel())
        self.previous[name] = (x.clone(),out.clone())


@torch.no_grad()
def run(args):
    base = args.long_base
    source,cfg,*_ = prepare_stream(base/'inputs',base/'clips')
    source.eval().requires_grad_(False)
    contract = read(args.cross/'contract.json')
    assert contract['group']=='original' and contract['plan']==digest(args.plan)
    assert contract['frozen_source']==read(base/'inputs/r297.json')['source_frozen_digest']
    plan = read(args.plan)
    rows = [r for r in plan['rows']['fit'] if r['scene'] in contract['scenes']]
    assert len(contract['scenes'])==8 and all(r['split']=='train' for r in rows)
    for row in rows:
        assert digest(Path(row['pcd']))==plan['members'][row['pcd']]
    # Only the frame ID and refresh reason enter the engineering experiment.
    with (args.cross/'frames.jsonl').open() as file:
        trace = [(r['frame'],r['refresh']) for r in map(json.loads,file)]
    assert [s for s,_ in trace]==[r['frame'] for r in rows]
    _,stream = loader(rows,cfg,0,303)
    audit = ObservationAudit()
    handles = [layer.register_forward_hook(partial(audit.record,name))
               for name,layer in source.core.observation.named_modules() if isinstance(layer,nn.Conv2d)]
    assert handles
    write_json(args.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        plan=contract['plan'], frozen_source=contract['frozen_source'], trace=digest(args.cross/'frames.jsonl'),
        scenes=contract['scenes'], grid='fixed ego grid', scoring=False, local_execution=False))
    with (args.output/'frames.jsonl').open('x') as file:
        for row,(_,reason),sample in zip(rows,trace,stream):
            assert row['frame']==sample['meta']['frame']
            if reason is not None:
                audit.previous.clear()
                check_next = True
            audit.verify_windows = reason is None and check_next
            audit.stats = {}
            ego = to_device(sample['ego'],torch.device('cuda'))
            source.core.observation(spatial_input(source,ego))
            assert len(audit.stats)==len(handles)
            file.write(json.dumps(dict(frame=row['frame'],scene=row['scene'],refresh=reason,
                layers=audit.stats,windows_checked=audit.verify_windows))+'\n')
            if audit.verify_windows:
                check_next = False
    for hook in handles:
        hook.remove()
    write_json(args.output/'report.json',dict(status='COMPLETED',scans=len(rows),layers=len(handles),
        optimizer_steps=0,detection_scores=0))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for key in ('long-base','plan','cross','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args)
