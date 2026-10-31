"""Check matched initial logits, local count provenance and actual new updates."""
import argparse
from pathlib import Path
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_event_quota import run as quota_audit, no_codec
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.train_e3dsnn_single import checked_forward, write_json
from opencood.data_utils.datasets import build_dataset


def run(a):
    seed();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3)
    cfg=load_yaml(a.config);ref_cfg=load_yaml(a.reference_config)
    model=train_utils.create_model(cfg).cuda();ref=train_utils.create_model(ref_cfg).cuda()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu'),strict=True)
    ref.load_state_dict(torch.load(a.reference_checkpoint,map_location='cpu'),strict=True)
    val=build_dataset(cfg,train=False);model.eval();ref.eval()
    assert cfg['postprocess']['anchor_args']['feature_stride']==4
    assert cfg['model']['args']['snapshot']==ref_cfg['model']['args']['snapshot']
    assert not ref.use_shallow_skip
    captured=[]
    handle=None
    if model.use_shallow_skip:
        def observe(module, inputs):
            x=inputs[0]
            assert x.min()>=0 and x.max()<=4 and torch.equal(x,x.round())
            captured.append(tuple(x.shape))
        handle=model.local_detail.register_forward_pre_hook(observe)
    for i in (0,1):
        batch=train_utils.to_device(val.collate_batch_test([val[i]]),torch.device('cuda'))
        with no_codec(),torch.no_grad():
            output=model(batch['ego']);control=ref(batch['ego'])
        assert all(torch.equal(output[k],control[k]) for k in ('psm','rm','dm'))
        assert output['psm'].shape[-2:]==batch['ego']['label_dict']['pos_equal_one'].shape[1:3]
        assert model._local_shallow is None
    if handle:handle.remove()
    del ref
    # Nonzero final projection exposes errors in per-sample routing and road
    # capture which an initial zero output alone would conceal.
    if model.use_shallow_skip:
        with torch.no_grad():model.local_detail.output.weight.normal_(0,.005)
        batch=train_utils.to_device(val.collate_batch_train([val[0],val[1]]),torch.device('cuda'))
        with no_codec(),torch.no_grad():batched=model(batch['ego'])
        for i in (0,1):
            single=train_utils.to_device(val.collate_batch_test([val[i]]),torch.device('cuda'))
            with no_codec(),torch.no_grad():alone=model(single['ego'])
            for key in ('psm','rm','dm'):
                torch.testing.assert_close(batched[key][i:i+1],alone[key],rtol=1e-3,atol=1e-4)
        train=build_dataset(cfg,train=True);seed();model.train()
        batch=train_utils.to_device(train.collate_batch_train([train[0],train[1]]),torch.device('cuda'))
        criterion=train_utils.create_loss(cfg).cuda()
        with no_codec():
            output,loss=checked_forward(model,criterion,batch);loss.backward()
        assert all(p.requires_grad for p in model.local_detail.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.local_detail.parameters())
    del model;torch.cuda.empty_cache()
    quota_audit(a)
    write_json(Path(a.output).with_name('highres_admission.json'),dict(passed=True,
        initial_logits_equal_interpolation_control=True,head_stride=4,local_count_shapes=captured,
        per_sample_routing_with_nonzero_projection='checked for shallow arm',
        no_continuous_shallow_bypass=True,audit_updates_discarded=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','checkpoint','reference-config','reference-checkpoint','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
