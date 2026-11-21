"""Measure causal local translation and phase-dependent decay on training middle BEV levels."""
import argparse
from collections import deque
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.calibrate_int_middle_budget import detector, training_rows
from opencood.tools.check_int_deep_spike_stream import build_source, read
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_membrane import load_quantum
from opencood.tools.eval_int_middle_cap import TRAIN_COUNTS
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


OFFSETS = [(0,0)] + [(y,x) for y in (-1,0,1) for x in (-1,0,1) if (y,x) != (0,0)]
GAINS = (1., .875, .75, .5)
TILE = 4


def tile_error(a, b, valid):
    error = ((a-b).abs().sum(1, keepdim=True) * valid).float()
    h,w = a.shape[-2:]
    padded = F.pad(error,(0,(-w)%TILE,0,(-h)%TILE))
    return F.avg_pool2d(padded,TILE)*TILE**2


def shifted(value, offset):
    return torch.roll(value, shifts=offset, dims=(-2,-1))


def select_levels(level, selected, valid):
    h,w = level.shape[-2:]
    choice = selected.repeat_interleave(TILE,-2).repeat_interleave(TILE,-1)[...,:h,:w]
    choice = torch.where(valid,choice,0)
    offset = torch.tensor(OFFSETS,device=level.device)[choice.flatten()]
    cells = torch.arange(h*w,device=level.device)
    index = cells - offset[:,0]*w - offset[:,1]
    return level.flatten(2).index_select(2,index).view_as(level), int(choice.count_nonzero())


def shuffled(selected, frame):
    flat = selected.flatten().clone()
    ids = flat.nonzero().flatten()
    if len(ids)>1:
        # Deterministic permutation among selected tiles; leave stationary tiles untouched.
        generator = torch.Generator().manual_seed(303+int(frame))
        perm = torch.randperm(len(ids),generator=generator).to(ids.device)
        flat[ids] = flat[ids[perm]]
    return flat.view_as(selected)


def metrics(target, predicted, groups):
    delta=(target-predicted).abs()
    u=delta.sum(1)
    n=(delta!=0).sum(1)
    return [[int(u[:,m].sum(dtype=torch.float64)),int(n[:,m].sum())] for m in groups]


def occupancy(ego, level):
    h,w=level.shape[-2:]
    coords=ego['processed_lidar']['voxel_coords']
    mask=torch.zeros_like(level[:,:1])
    # Nearest middle lattice centre for each occupied input pillar.
    y=((coords[:,-2].long()+2)//4).clamp(max=h-1)
    x=((coords[:,-1].long()+2)//4).clamp(max=w-1)
    mask[0,0,y,x]=1
    return mask


def probe(history, target, meta, area):
    last = history[-1]
    def align(value, past):
        return warp_bev(value,past['pose'],meta['pose'],area)[0]
    packed = align(torch.cat([last['levels'],last['occupied'],torch.ones_like(last['occupied'])],1),last)
    base, occupied, known = packed[:,:-2],packed[:,-2:-1].bool(),packed[:,-1:].bool()
    groups = [(known & occupied)[0,0],(known & ~occupied)[0,0],(~known)[0,0]]
    baseline = metrics(target,base,groups)
    rec=dict(regions=[int(m.sum()) for m in groups], zero=baseline, eligible=0,
             moved={}, searched=False,shape=list(base.shape),decay={})
    for gain in GAINS:
        rec['decay'][str(gain)] = baseline if gain == 1 else metrics(target,(base*gain).round(),groups)
    if len(history)==2:
        older=history[0]
        dt = meta['timestamp_us']-last['stamp']
        prev_dt = last['stamp']-older['stamp']
        rec['searched'] = prev_dt>0 and .75 <= dt/prev_dt <= 1.25
    if not rec['searched']:
        for name in ('causal','shuffle','oracle'):
            rec[name]=baseline
            rec['moved'][name]=0
        return rec
    desc=align(torch.cat([older['levels'][:,::32],torch.ones_like(older['occupied'])],1),older)
    # Common support for every candidate, including zero, excludes wrap and holes.
    invalid=~(known & desc[:,-1:].bool())
    valid=F.max_pool2d(F.pad(invalid.float(),(1,1,1,1),value=1.),3,1)==0
    rec['eligible']=int(valid.sum())
    assert desc.shape[1]==9
    past_costs=[]
    oracle_costs=[]
    for offset in OFFSETS:
        past_costs.append(tile_error(base[:,::32],shifted(desc[:,:-1],offset),valid))
        oracle_costs.append(tile_error(target,shifted(base,offset),valid))
    causal=torch.stack(past_costs).argmin(0)
    oracle=torch.stack(oracle_costs).argmin(0)
    for name,choice in [('causal',causal),('shuffle',shuffled(causal,meta['frame'])),('oracle',oracle)]:
        predicted,rec['moved'][name]=select_levels(base,choice,valid)
        rec[name]=metrics(target,predicted,groups)
    return rec


@torch.no_grad()
def run(inputs, training, counts, out):
    torch.manual_seed(303)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    ref=read(inputs/'r297.json')
    for name,key in [('plan.json','plan'),('traces.json','traces'),('quantum.json','calibration')]:
        assert digest(inputs/name)==ref['inputs_sha256'][key]
    assert digest(counts)==TRAIN_COUNTS
    reference=[json.loads(line) for line in counts.read_text().splitlines()]
    plan=read(inputs/'plan.json')
    rows,traces,starts=training_rows(inputs,training,plan)
    cfg=load_yaml(str(inputs/'config.yaml'))
    cfg['data_augment']=[]
    source=build_source(inputs,cfg)
    assert model_digest(source)==ref['source_frozen_digest']
    q,_,_=load_quantum(inputs/'quantum.json',plan,512)
    model=detector(source,q,'carry',.25)
    _,stream=loader(rows,cfg,0,303)
    write_json(out/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        train_plan=digest(training/'train_plan.json'), inputs={f:digest(inputs/f) for f in
        ('parent.pth','protected_light_final.pth','readout.pth','traces.json','quantum.json','config.yaml')},
        source=ref['source_frozen_digest'], cal=TRAIN_COUNTS, region_order=['last_occupied','last_empty','new'],
        metrics=['units','addresses'], offsets=OFFSETS, gains=GAINS, descriptor_channels=list(range(0,256,32)),
        seed=303, tile=TILE, labels_used=0,quantum=.25,device=torch.cuda.get_device_name(),
        params=sum(p.numel() for p in source.parameters()),torch=torch.__version__))
    n=0
    history=deque(maxlen=2)
    with (out/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace,check in zip(stream,traces,reference):
            meta=sample['meta']
            assert meta['frame']==trace['frame']==check['frame'] and not meta['supervised']
            if meta['frame'] in starts:
                state=None
                history.clear()
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            _,state,info=model.step_spatial(spatial,state,meta,local_request(trace_masks(trace,spatial.device)))
            level=torch.round(state['coarse']/.25)
            rec=dict(frame=meta['frame'],scene=meta['scene'],refresh=trace['refresh'])
            assert info['full_refresh']==rec['refresh']==check['refresh']
            if rec['refresh'] is None:
                age+=1
                area=block_lidar_range(source.bounds,spatial.shape[-2:],level.shape[-2:],4)
                rec.update(probe(history,level,meta,area),age=age)
                units,addresses=map(sum,zip(*rec['zero']))
                assert [addresses,units]==check['carry'][0], rec['frame']
                assert sum(v[0] for v in rec['oracle'])<=units
            else:
                age=0
            history.append(dict(levels=level,pose=meta['pose'],stamp=meta['timestamp_us'],occupied=occupancy(ego,level)))
            log.write(json.dumps(rec)+'\n')
            n+=1
    assert n==len(rows)==len(traces)==len(reference)
    write_json(out/'report.json',dict(scans=n,status='COMPLETED',zero_replay_exact=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('inputs','training','counts','output'):
        parser.add_argument('--'+arg,required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args.inputs,args.training,args.counts,args.output)
