"""Read-only fixed-source current/range audit; no AP fitting or new training."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from opencood.tools.launch_e3dsnn_hybrid_round import sha
from opencood.tools.train_e3dsnn_single import write_json

SOURCE=Path('/data1/tzm/workspace/diagnostics/e3dsnn_snapshot_adapt_20261021/if8')
DIGEST='c51a33f3c1d935a8528763ddbb68ccb3298475beab842a49c5a71216c56b5a78'


def measure(local,current,mask):
    """All channels at selected spatial sites; distinguish clipping and rounding."""
    a=local[:,mask].astype(np.float64);b=current[:,mask].astype(np.float64)
    if not a.size:return dict(sites=0,values=0)
    u=a+b;q=np.floor(np.clip(u,0,4)+.5);ego=np.floor(np.clip(a,0,4)+.5)
    return dict(sites=int(mask.sum()),values=int(a.size),local_rms=float(np.sqrt(np.mean(a*a))),
        current_rms=float(np.sqrt(np.mean(b*b))),current_mean=float(b.mean()),
        current_negative_fraction=float(np.mean(b<0)),
        local_above4_fraction=float(np.mean(a>4)),combined_above4_fraction=float(np.mean(u>4)),
        newly_above4_fraction=float(np.mean((a<=4)&(u>4))),
        local_below0_fraction=float(np.mean(a<0)),combined_below0_fraction=float(np.mean(u<0)),
        newly_below0_fraction=float(np.mean((a>=0)&(u<0))),
        clipped_charge_abs_mean=float(np.mean(np.abs(u-np.clip(u,0,4)))),
        output_four_fraction=float(np.mean(q==4)),output_zero_fraction=float(np.mean(q==0)),
        increased_code_fraction=float(np.mean(q>ego)),decreased_code_fraction=float(np.mean(q<ego)))


def run(root):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.models.fuse_modules import e3dsnn_snapshot_adapter as adapter
    from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume
    from opencood.models.fuse_modules.e3dsnn_progressive import count_readout
    checkpoint=SOURCE/'train/best_ap70.pth';assert sha(checkpoint)==DIGEST
    root.mkdir(parents=True,exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(.2)
    random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    cfg=load_yaml(str(SOURCE/'train.yaml'));data=build_dataset(cfg,train=False)
    assert len(data)==1738
    model=train_utils.create_model(cfg).cuda().eval();initial=torch.load(str(checkpoint),map_location='cpu')
    model.load_state_dict(initial,strict=True);model.requires_grad_(False);model.packet_roundtrip=True;model.snapshot_audit=True
    native=adapter.stream_readout;captured={}
    def observed(membrane,counts,pose,contract,receiver,decode):
        aligned,coverage=warp_volume(counts,pose,contract.origin_xyz,contract.step_xyz,geometry_dtype=torch.float64)
        current=decode(aligned)*(2*torch.sigmoid(receiver.local_gate(membrane))*receiver.current_scale)
        supported=(aligned.detach()!=0).any(1,keepdim=True)&(coverage>0)
        expected=torch.where(supported,count_readout(membrane+current),count_readout(membrane))
        result=native(membrane,counts,pose,contract,receiver,decode)
        assert torch.equal(expected,result)
        captured.update(local=membrane[0].cpu().numpy(),current=current[0].cpu().numpy(),
            support=supported[0,0].cpu().numpy(),counts=counts[0].cpu().numpy())
        return result
    indices=np.linspace(0,len(data)-1,64,dtype=np.int64).tolist()
    protocol=dict(checkpoint=str(checkpoint),checkpoint_sha256=DIGEST,indices=indices,budgets=[2048,4096,None],
        seed=20260929,config_sha256=sha(SOURCE/'train.yaml'),training=False,
        source='each frame encoded once; exact same cached local/road tensors for all three budgets',
        masks='own support, fixed 2048 support, and local-active part of fixed support',
        no_AP=True,limitation='systematic 64-frame descriptive diagnostic; not causal evidence or representative AP',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip())
    write_json(root/'protocol.json',protocol)
    rows=[];adapter.stream_readout=observed;started=time.perf_counter()
    try:
        with torch.no_grad():
            for index in indices:
                batch=train_utils.to_device(data.collate_batch_test([data[index]]),torch.device('cuda'));item=batch['ego']
                local_fn,road_fn=model.encode_membrane,model.encode_lidar
                local,membrane=local_fn(item['processed_lidar'],1);road=road_fn(item['road_lidar'],1)
                model.encode_membrane=lambda lidar,n:(local,membrane);model.encode_lidar=lambda lidar,n:road
                variants=[];previous=None;fixed=None;local_active=None
                try:
                    for budget in (2048,4096,None):
                        model.snapshot_budget=budget;output=model(item);obs=dict(captured)
                        if previous is not None:assert np.all(obs['counts']>=previous)
                        previous=obs['counts']
                        if fixed is None:
                            fixed=obs['support'].copy();local_active=fixed & np.any(obs['local']!=0,axis=0)
                        assert np.all(~fixed|obs['support'])
                        variants.append(dict(budget_bytes=budget,actual_bytes=output['packet_bytes'][0],
                            source_count_sum=float(obs['counts'].sum()),
                            source_nonzero_channels=int(np.count_nonzero(obs['counts'])),
                            own_support=measure(obs['local'],obs['current'],obs['support']),
                            fixed_2048_support=measure(obs['local'],obs['current'],fixed),
                            fixed_local_active_support=measure(obs['local'],obs['current'],local_active),
                            packet_sha256=model.snapshot_records[0]['packet_sha256'],exact_original_readout=True))
                finally:model.encode_membrane,model.encode_lidar=local_fn,road_fn
                rows.append(dict(index=index,frame_id=item['frame_ids'][0],variants=variants))
                write_json(root/'progress.json',dict(frames=len(rows),total=64,elapsed_seconds=time.perf_counter()-started))
        assert sha(checkpoint)==DIGEST
        assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
        means={}
        for i,label in enumerate(('b2048','b4096','full')):
            means[label]={}
            for region in ('own_support','fixed_2048_support','fixed_local_active_support'):
                entries=[row['variants'][i][region] for row in rows if row['variants'][i][region]['values']]
                means[label][region]={key:float(np.mean([v[key] for v in entries])) for key in entries[0]} if entries else {}
        write_json(root/'summary.json',dict(passed=True,frames=64,all_state_unchanged=True,protocol=protocol,
            averaging='unweighted per-frame means, excluding empty masks; not pooled voxel statistics',means=means,records=rows))
        write_json(root/'status.json',dict(state='complete',summary_sha256=sha(root/'summary.json')))
        print(json.dumps(dict(passed=True,frames=64,means=means)),flush=True)
    finally:adapter.stream_readout=native


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(Path(a.root))
