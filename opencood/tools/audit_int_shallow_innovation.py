"""Independently account for unserved shallow innovations before a detection decision."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.train_int_silent_task import (
    digest, loader, prepare_stream, read, spatial_input, to_device, write_json)


def blocks(innovation):
    h,w=innovation.shape[-2:]
    ph,pw=(-h)%4,(-w)%4
    energy=innovation.abs().long().sum(1)
    def tile_sum(grid):
        padded=F.pad(grid,(0,pw,0,ph))
        return padded.reshape((h+ph)//4,4,(w+pw)//4,4).sum((1,3))
    energies=tile_sum(energy).cpu().numpy()
    areas=tile_sum(torch.ones_like(energy)).cpu().numpy()
    scores=(energies/areas).flatten()
    ids=np.lexsort((np.arange(len(scores)),-scores))
    selected=np.zeros(len(scores),dtype=np.bool_)
    selected[ids[:(len(scores)+1)//2]]=True
    support=hashlib.sha256(selected.tobytes()).hexdigest()
    tiles=torch.as_tensor(selected.reshape(energies.shape),device=energy.device)
    return tiles.repeat_interleave(4,0).repeat_interleave(4,1)[:h,:w],support


@torch.no_grad()
def run(a):
    c=read(a.exact/'contract.json')
    p=read(a.plan)
    assert digest(a.plan)==c['parent']['plan']
    assert read(a.long_base/'inputs/r297.json')['source_frozen_digest']==c['parent']['frozen_source']
    rows=[r for r in p['rows']['fit'] if r['scene'] in c['scenes']]
    trace=[json.loads(l) for l in (a.exact/'frames.jsonl').read_bytes().splitlines()]
    assert [r['frame'] for r in rows]==[r['frame'] for r in trace]
    for r in rows:assert r['split']=='train' and digest(Path(r['pcd']))==p['members'][r['pcd']]
    source,cfg,*_=prepare_stream(a.long_base/'inputs',a.long_base/'clips')
    source.eval().requires_grad_(False)
    _,stream=loader(rows,cfg,0,303)
    block=source.core.observation[0]
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        plan=c['parent']['plan'],scenes=c['scenes'],source=c['parent']['frozen_source'],scoring=False))
    with (a.output/'frames.jsonl').open('x') as f:
        for i,(sample,r) in enumerate(zip(stream,trace)):
            meta=sample['meta']
            assert r['frame']==meta['frame']
            spatial=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
            observed=(4*block.relu(block.bn1(block.conv1(spatial)))).round().to(torch.int32)
            item=dict(frame=meta['frame'],scene=meta['scene'],refresh=r['refresh'],arms={})
            if r['refresh'] is not None:
                age=0
                states={k:observed for k in ('unit','amplitude')}
            else:
                age+=1
                area=block_lidar_range(source.bounds,spatial.shape[2:],observed.shape[2:],2)
                def predict(value):
                    return warp_bev(value.float(),pose,meta['pose'],area)[0].int()
                allowed,item['support']=blocks(observed-predict(last_observation))
                for name,old in states.items():
                    prior=predict(old)
                    desired=observed-prior
                    offered=desired.masked_fill(~allowed,0)
                    pulse=offered.sign() if name=='unit' else offered
                    updated=prior+pulse
                    states[name]=updated
                    abs_offer=offered.abs()
                    demand=int(desired.abs().sum())
                    inside=int(abs_offer.sum())
                    sent=int(pulse.abs().sum())
                    residual=int((observed-updated).abs().sum())
                    assert demand==sent+residual and inside>=sent
                    item['arms'][name]=dict(demand=demand,blocked_by_tile=demand-inside,
                        blocked_by_magnitude=inside-sent,sent_units=sent,sent_addresses=int(pulse.count_nonzero()),
                        actual_changes=int((updated!=old).sum()),residual=residual,allowed=int(allowed.sum())*128,
                        offered_multi=int((abs_offer>1).sum()),offered_ge4=int((abs_offer>=4).sum()),
                        offered_ge8=int((abs_offer>=8).sum()))
            pose,last_observation=meta['pose'],observed
            item['age']=age
            f.write(json.dumps(item)+'\n')
    assert i+1==len(rows)
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=i+1,optimizer_steps=0,detections=0))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('plan','exact','long-base','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)
