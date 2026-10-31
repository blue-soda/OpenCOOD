"""Frozen training-frame checks for local Count4 detail; no AP or wire work."""
import argparse
import hashlib
from pathlib import Path
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_event_quota import no_codec
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.train_e3dsnn_single import write_json
from opencood.data_utils.datasets import build_dataset


def run(a):
    seed();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3)
    cfg=load_yaml(a.config);data=build_dataset(cfg,train=True)
    model=train_utils.create_model(cfg).cuda()
    state=torch.load(a.checkpoint,map_location='cpu');model.load_state_dict(state,strict=True);model.eval()
    assert model.use_shallow_skip
    frames=[]
    for i in range(a.frames):
        batch=train_utils.to_device(data.collate_batch_test([data[i]]),torch.device('cuda'))
        captured={}
        def observe(module,inputs,output):
            captured['input']=inputs[0].detach();captured['delta']=output.detach()
        handle=model.local_detail.register_forward_hook(observe)
        try:
            with no_codec(),torch.no_grad():full=model(batch['ego'])
        finally:handle.remove()
        handle=model.local_detail.register_forward_hook(lambda m,x,y:torch.zeros_like(y))
        try:
            with no_codec(),torch.no_grad():without=model(batch['ego'])
        finally:handle.remove()
        delta=captured['delta'];counts=captured['input']
        assert torch.isfinite(delta).all() and delta.abs().max()<=1.00001
        assert counts.min()>=0 and counts.max()<=4 and torch.equal(counts,counts.round())
        frames.append(dict(index=i,delta_abs_mean=delta.abs().mean().item(),delta_abs_max=delta.abs().max().item(),
            count_nonzero_fraction=(counts!=0).float().mean().item(),
            cls_abs_change=(full['psm']-without['psm']).abs().mean().item(),
            regression_abs_change=(full['rm']-without['rm']).abs().mean().item()))
    assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items())
    report=dict(passed=True,frames=frames,means={k:sum(v[k] for v in frames)/len(frames) for k in frames[0] if k!='index'},
        config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        model_state_unchanged=True,no_optimization=True,no_ap_selection=True,no_wire_calls=True,
        split='first training entries, seed20260929 cached per-frame augmentation',bound=1.)
    write_json(Path(a.output),report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--frames',type=int,default=32)
    run(p.parse_args())
