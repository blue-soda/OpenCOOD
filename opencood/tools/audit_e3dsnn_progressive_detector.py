"""Audit progressive detection integration on a true augmented batch of two."""
import argparse
import json
from pathlib import Path
import time
import random
import numpy as np
import torch
import spconv.pytorch as spconv
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import checked_forward
from opencood.tools.launch_e3dsnn_hybrid_round import sha,read
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.utils.e3dsnn_budget import decoder_digest
from opencood.utils.e3dsnn_fusion_probe import bn_digest


def audit(root,base):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError('Fresh audit directory required')
    source=Path(base)/'e3dsnn_budget_round1_20261018'
    prior=read(source/'adapt10_summary.json');assert prior['passed']
    checkpoint=source/'budget_s_fusion_adapt10/best_ap70.pth'
    assert sha(checkpoint)==prior['groups']['s_fusion']['checkpoint_sha256']
    cfg=load_yaml(str(source/'s_fusion.yaml'))
    cfg['model']['core_method']='e3dsnn_progressive_detector'
    cfg['model']['args'].update(progressive_packets=3,readout_prefix=3,
        adaptive_membrane_gate=True,retain_membrane_residual=True)
    random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929)
    dataset=build_dataset(cfg,train=True)
    samples=[dataset[i] for i in (0,1)]
    batch=train_utils.to_device(dataset.collate_batch_train(samples),torch.device('cuda'))
    data=batch['ego'];model=train_utils.create_model(cfg).cuda().eval()
    model.load_legacy(torch.load(str(checkpoint),map_location='cpu'))
    report=dict(started_unix=time.time(),source_sha256=sha(checkpoint),frames=data['frame_ids'],
        road_frames=data['road_frame_ids'],prefixes=[],scope='Augmented train frames 0/1; discarded optimization step; no AP estimate.')
    encode_membrane=model.encode_membrane;encode=model.encode_lidar
    with torch.no_grad():
        ego,membrane=encode_membrane(data['processed_lidar'],2);road=encode(data['road_lidar'],2)
        model.encode_membrane=lambda lidar,n:(ego,membrane)
        model.encode_lidar=lambda lidar,n:road
        for prefix in (1,2,3):
            model.readout_prefix=prefix;model.packet_roundtrip=False;direct=model(data)
            model.packet_roundtrip=True;wire=model(data)
            assert all(torch.equal(direct[k],wire[k]) for k in ('psm','rm','dm'))
            assert len(wire['packet_bytes'])==2
            report['prefixes'].append(dict(prefix=prefix,packet_bytes=wire['packet_bytes'],direct_wire_exact=True))
        repeated=model(data)
        assert all(torch.equal(repeated[k],wire[k]) for k in ('psm','rm','dm'))
        report['repeated_forward_has_no_state_leak']=True
        # Same cached encoder tensors, isolated per-frame execution: no cross-batch state mixing.
        model.packet_roundtrip=False;combined=model(data);errors=[]
        for i in range(2):
            one=train_utils.to_device(dataset.collate_batch_train([samples[i]]),torch.device('cuda'))['ego']
            active=road.indices[:,0]==i;indices=road.indices[active].clone();indices[:,0]=0
            subset=spconv.SparseConvTensor(road.features[active],indices,road.spatial_shape,1)
            model.encode_membrane=lambda lidar,n:(None,membrane[i:i+1])
            model.encode_lidar=lambda lidar,n:subset
            isolated=model(one)
            for key in ('psm','rm','dm'):
                error=float((combined[key][i:i+1]-isolated[key]).abs().max());errors.append(error)
                assert torch.allclose(combined[key][i:i+1],isolated[key],atol=2e-5,rtol=2e-5)
        report['batch_isolation_max_response_error']=max(errors)
    model.encode_membrane=encode_membrane;model.encode_lidar=encode
    model.packet_roundtrip=False;model.readout_prefix=3;model.train()
    enc,dec,bn=encoder_digest(model),decoder_digest(model),bn_digest(model)
    criterion=train_utils.create_loss(cfg).cuda();optimizer=train_utils.setup_optimizer(cfg,model)
    model.message_dropout=0.;optimizer.zero_grad(set_to_none=True)
    output,loss=checked_forward(model,criterion,batch)
    assert len(output['prefix_outputs'])==3
    loss.backward()
    grads={k:float(p.grad.abs().sum()) for k,p in model.named_parameters() if p.grad is not None}
    assert set(grads)=={'membrane_fusion.basis.weight','membrane_fusion.gate.weight'}
    assert all(v>0 and np.isfinite(v) for v in grads.values())
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
    assert encoder_digest(model)==enc and decoder_digest(model)==dec and bn_digest(model)==bn
    assert not any('residual' in k or 'next_packet' in k for k in model.state_dict())
    report.update(gradient_l1=grads,loss_components=model.last_loss_components,
        frozen_encoder_decoder_bn=True,states_not_checkpointed=True)
    model.message_dropout=1.
    with torch.no_grad():
        dropped=model(data)
        for prefix in dropped['prefix_outputs']:
            assert all(torch.equal(prefix[k],dropped['ego_output'][k]) for k in ('psm','rm','dm'))
        assert torch.equal(dropped['message_kept'],torch.zeros_like(dropped['message_kept']))
    report['full_dropout_all_prefixes_equal_local_readout']=True
    report['passed']=True;report['finished_unix']=time.time()
    (root/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--base',default='/data1/tzm/workspace/diagnostics')
    args=parser.parse_args();audit(args.root,args.base)
