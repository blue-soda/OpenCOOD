"""Read-only training-frame diagnostics of the added current before Count4.

No optimization, AP selection, or serialization; compare frozen initial and
trained snapshots on exactly the same seeded training inputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.audit_snapshot_event_quota import no_codec
from opencood.tools.train_e3dsnn_single import write_json
from opencood.models.fuse_modules.e3dsnn_pyramid_current import support_block
from opencood.models.fuse_modules.e3dsnn_progressive import count_readout


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    seed();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3)
    cfg=load_yaml(a.config);dataset=build_dataset(cfg,train=True)
    assert len(dataset)==4678 and a.frames==32
    model=train_utils.create_model(cfg).cuda().eval()
    assert model.snapshot_selection=='event_quota' and hasattr(model.joint_receiver,'snapshot_pyramid')
    model.packet_roundtrip=False;model.snapshot_measure_bytes=False;model.snapshot_packet_directory=None
    # Cache the augmented CPU batches once, independently of model state.
    seed();batches=[dataset.collate_batch_train([dataset[i]]) for i in range(a.frames)]
    pyramid=model.joint_receiver.snapshot_pyramid;stages=[[],[],[]];observed=[]
    def stage_hook(level):
        def hook(module,args,result):stages[level].append(result.detach())
        return hook
    handles=[stage.register_forward_hook(stage_hook(i)) for i,stage in enumerate(pyramid.blocks)]
    def inspect(module,args,result):
        membrane,decoded,base,received,coverage=args;current,supported=result
        delta=current-base*received.to(base.dtype)
        local=count_readout(membrane)
        before=torch.where(received,count_readout(membrane+base),local)
        after=torch.where(supported,count_readout(membrane+current),local)
        expanded=supported.expand_as(membrane);changed=before!=after
        def fraction(x):return float(x.float().mean())
        row=dict(delta_abs_mean=float(delta.abs().mean()),base_abs_mean=float(base.abs().mean()),
            delta_abs_max=float(delta.abs().max()),delta_to_base_l2=float(delta.norm()/base.norm().clamp_min(1e-12)),
            received_fraction=fraction(received),supported_fraction=fraction(supported),
            count_changed_global=fraction(changed),
            count_changed_supported=float((changed & expanded).sum()/expanded.sum().clamp_min(1).float()),
            local_zero=fraction(local==0),local_four=fraction(local==4),
            fused_zero=fraction(after==0),fused_four=fraction(after==4),attention=[])
        mask=F.max_pool3d(F.max_pool3d(received.float(),3,1,1),3,1,1).amax(2)
        for level,pair in enumerate(stages):
            assert len(pair)==2
            local_feature,remote_feature=pair;mask=support_block(mask,1 if level==0 else 2)
            remote_feature=remote_feature*mask
            scores=torch.stack(((local_feature*local_feature).sum(1),(local_feature*remote_feature).sum(1)),1)/math.sqrt(local_feature.shape[1])
            scores[:,1:2]=scores[:,1:2].masked_fill(mask==0,-1e4)
            weights=scores.softmax(1)[:,1:2][mask>0]
            row['attention'].append(dict(level=level,remote_weight_mean=float(weights.mean()) if weights.numel() else 0.,
                remote_below_one_percent=fraction(weights<.01) if weights.numel() else 0.))
        observed.append(row)
    handles.append(pyramid.register_forward_hook(inspect))
    report=dict(config_sha256=sha(a.config),frames=a.frames,split='first 32 training entries, seed20260929 cached augmentation',
        no_optimization=True,no_ap_selection=True,no_wire_calls=True,states={})
    for label,path in [('initial',a.initial),('trained',a.checkpoint)]:
        state=torch.load(path,map_location='cpu');model.load_state_dict(state,strict=True)
        rows=[]
        for index,batch_cpu in enumerate(batches):
            for items in stages:items.clear()
            batch=train_utils.to_device(batch_cpu,torch.device('cuda'))
            with no_codec(),torch.no_grad():output=model(batch['ego'])
            assert all(torch.isfinite(output[k]).all() for k in ('psm','rm','dm'))
            assert 'packet_bytes' not in output
            row=observed.pop();row.update(index=index,frame_id=str(batch_cpu['ego']['frame_ids'][0]),events=output['stream_events'][0]);rows.append(row)
            if getattr(pyramid,'residual_limit',None) is not None:
                assert row['delta_abs_max'] <= pyramid.residual_limit+1e-5
            if label=='initial':assert row['delta_abs_max']==0 and row['count_changed_global']==0
            if (index+1)%8==0:print(json.dumps(dict(state=label,frames=index+1)),flush=True)
        assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items())
        averages={k:float(np.mean([row[k] for row in rows])) for k in rows[0] if k not in ('attention','index','frame_id')}
        averages['attention']=[{k:float(np.mean([row['attention'][i][k] for row in rows])) for k in ('remote_weight_mean','remote_below_one_percent')} for i in range(3)]
        report['states'][label]=dict(checkpoint_sha256=sha(path),means=averages,frames=rows,
            output_weight_norm=float(pyramid.output.weight.norm()))
    for h in handles:h.remove()
    report.update(passed=True,residual_limit=getattr(pyramid,'residual_limit',None),
                  normalized_merge=getattr(pyramid,'normalize_merge',False))
    write_json(Path(a.output),report)
    print(json.dumps({key:value['means'] for key,value in report['states'].items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','initial','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--frames',type=int,default=32)
    run(p.parse_args())
