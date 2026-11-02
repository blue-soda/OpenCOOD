"""Real-frame admission for fixed-sender, matched receiver adaptation."""
import argparse, copy, hashlib
from pathlib import Path
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.audit_snapshot_event_quota import no_codec
from opencood.tools.train_e3dsnn_single import checked_forward, write_json
from opencood.utils.cp_optimizer import make_optimizer
from opencood.models.fuse_modules.e3dsnn_receiver_probe import self_test


def run(a):
    self_test(); seed(); torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35)
    cfg=load_yaml(a.config); initial=torch.load(a.checkpoint,map_location='cpu')
    model=train_utils.create_model(cfg).cuda(); model.load_state_dict(initial,strict=True)
    reference_cfg=copy.deepcopy(cfg);reference_cfg['model']['core_method']='e3dsnn_snapshot_highres'
    base=train_utils.create_model(reference_cfg).cuda();base.load_state_dict(initial,strict=True)
    model.eval();base.eval();val=build_dataset(cfg,train=False);records=[]
    for index in (0,1):
        batch=train_utils.to_device(val.collate_batch_test([val[index]]),torch.device('cuda'))
        for budget in (0,2304,8192):
            model.snapshot_budget=base.snapshot_budget=budget
            with no_codec(),torch.no_grad():
                direct=model(batch['ego']);original=base(batch['ego'])
            if budget==0 or model.receiver_probe=='membrane':
                assert all(torch.equal(direct[k],original[k]) for k in ('psm','rm','dm'))
            assert direct['stream_events']==original['stream_events']
            model.packet_roundtrip=True;model.snapshot_measure_bytes=True
            with torch.no_grad():measured=model(batch['ego'])
            assert all(torch.equal(direct[k],measured[k]) for k in ('psm','rm','dm'))
            assert measured['communication_measured'] and not direct['communication_measured']
            model.packet_roundtrip=False;model.snapshot_measure_bytes=False
            records.append(dict(index=index,budget=budget,events=direct['stream_events'],
                bytes=measured['packet_bytes'],wire_outputs_exact=True,
                source_predictions_exact=budget==0 or model.receiver_probe=='membrane'))
    # Check the sender drive itself, not just the fixed number of selected events.
    with no_codec(),torch.no_grad():
        road=batch['ego']['road_lidar']
        x=model.encode_lidar(road,1).dense();y=base.encode_lidar(road,1).dense()
        assert torch.equal(x,y)
        assert torch.equal(model.snapshot_drive(x),base.snapshot_drive(y))
    del base,x,y
    model.train();model.snapshot_budget=8192
    fixed=model.fixed_state_digest();optimizer=make_optimizer(cfg,model)
    criterion=train_utils.create_loss(cfg).cuda();train=build_dataset(cfg,train=True)
    seed();batch=train_utils.to_device(train.collate_batch_train([train[0],train[1]]),torch.device('cuda'))
    steps=[]
    for step in range(2):
        seed() # fixed stochastic choices for the two admission steps, not formal training
        optimizer.zero_grad(set_to_none=True)
        with no_codec():out,loss=checked_forward(model,criterion,batch);loss.backward()
        gradients={n:p.grad for n,p in model.named_parameters() if p.grad is not None}
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        assert not any(n.startswith(('backbone_3d.','content.project.')) for n in gradients)
        for prefix in ('content.expand.','joint_receiver.','local_detail.','backbone_2d.'):
            assert any(n.startswith(prefix) and g.abs().sum()>0 for n,g in gradients.items()),prefix
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(norm)
        optimizer.step();assert model.fixed_state_digest()==fixed
        assert not out['communication_measured'] and 'packet_bytes' not in out
        steps.append(dict(loss=float(loss),gradient_norm=float(norm),gradient_tensors=len(gradients)))
    changed=[n for n,v in model.state_dict().items() if not torch.equal(v.cpu(),initial[n])]
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert changed and set(changed)<=set(names)
    write_json(Path(a.output),dict(passed=True,mode=model.receiver_probe,frames=records,steps=steps,
        fixed_encoder_and_sender=True,frozen_state_unchanged=True,trainable=names,changed=changed,
        updates_discarded=True,no_training_codec=True,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('config','checkpoint','output'):parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
