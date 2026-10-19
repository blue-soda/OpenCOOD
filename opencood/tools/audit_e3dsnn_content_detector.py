"""Real augmented batch2 content protocol, source and gradient gate."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
import spconv.pytorch as spconv
import yaml
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import checked_forward
from opencood.tools.launch_e3dsnn_hybrid_round import sha, read
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.utils.e3dsnn_budget import decoder_digest
from opencood.utils.e3dsnn_fusion_probe import bn_digest


def audit(root, base, family, learn, joint=False, joint_variant='full'):
    if joint_variant not in ('full','no_local','clear_interbatch','frozen_order'):
        raise ValueError('Unknown registered joint ablation')
    if not joint and joint_variant!='full':raise ValueError('Ablations require joint detector')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError('Fresh audit directory required')
    if joint:
        if family!='snn' or not learn:raise ValueError('Joint audit requires SNN learned content')
        source=Path(base)/'e3dsnn_content_round1_v2_20261019';group='snn_learn'
        ckpt=source/'snn_learn_adapt10/best_ap70.pth'
    elif family=='snn':
        source=Path(base)/'e3dsnn_message_fusion_round1_20261018';group='snn_geometry'
        ckpt=source/'message_snn_geometry_adapt10/best_ap70.pth'
    else:
        source=Path(base)/'e3dsnn_budget_round1_20261018';group='a_fusion'
        ckpt=source/'budget_a_fusion_adapt10/best_ap70.pth'
    prior=read(source/'adapt10_summary.json');assert prior['passed']
    assert sha(ckpt)==prior['groups'][group]['checkpoint_sha256']
    source_config=source/'snn_learn/p3.yaml' if joint else source/(group+'.yaml')
    raw=yaml.safe_load(source_config.read_text())
    raw['model']['core_method']='e3dsnn_joint_detector' if joint else 'e3dsnn_content_detector'
    raw['model']['args'].update(content_trainable=learn,readout_prefix=3,content_full_packet=False,
        fusion_activation_kind='count4' if family=='snn' else 'relu',train_encoder=False)
    if joint:
        raw['model']['args'].update(joint_local_membrane=joint_variant!='no_local',
            joint_retain_residual=joint_variant!='clear_interbatch',
            joint_learn_order=joint_variant!='frozen_order')
    raw['train_params']=dict(batch_size=2,epoches=10)
    (root/'config.yaml').write_text(yaml.safe_dump(raw));cfg=load_yaml(str(root/'config.yaml'))
    random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929)
    dataset=build_dataset(cfg,train=True);samples=[dataset[i] for i in (0,1)]
    batch=train_utils.to_device(dataset.collate_batch_train(samples),torch.device('cuda'));data=batch['ego']
    model=train_utils.create_model(cfg).cuda().eval()
    (model.load_content if joint else model.load_legacy)(torch.load(str(ckpt),map_location='cpu'))
    torch.save(model.state_dict(),str(root/'initial.pth'))
    report=dict(family=family,learn_content=learn,joint_membrane=joint,started_unix=time.time(),source=str(ckpt),
        source_sha256=sha(ckpt),source_summary_sha256=sha(source/'adapt10_summary.json'),
        initial_sha256=sha(root/'initial.pth'),config_sha256=sha(root/'config.yaml'),
        frames=data['frame_ids'],road_frames=data['road_frame_ids'],prefixes=[],
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initial_basis_sha256=model.content.basis_digest())
    if joint:
        report['joint_variant']=joint_variant
        reference=Path(base)/'e3dsnn_joint_round1_20261019/joint/initial.pth'
        if reference.exists():
            expected=torch.load(str(reference),map_location='cpu');actual=model.state_dict()
            assert set(actual)==set(expected) and all(torch.equal(actual[k].cpu(),v) for k,v in expected.items())
            report['initial_tensors_equal_full_joint']=True
            report['full_joint_initial_sha256']=sha(reference)
    original=model.encode_lidar;original_membrane=model.encode_membrane
    with torch.no_grad():
        ego,membrane=original_membrane(data['processed_lidar'],2);road=original(data['road_lidar'],2)
        model.encode_membrane=lambda lidar,n:(ego,membrane);model.encode_lidar=lambda lidar,n:road
        for prefix in (1,2,3):
            model.readout_prefix=prefix;model.packet_roundtrip=False;direct=model(data)
            model.packet_roundtrip=True;wire=model(data)
            assert all(torch.equal(direct[k],wire[k]) for k in ('psm','rm','dm'))
            report['prefixes'].append(dict(prefix=prefix,packet_bytes=wire['packet_bytes'],direct_wire_exact=True))
        model.content_full_packet=True;full=model(data);model.content_full_packet=False
        assert all(torch.equal(full[k],wire[k]) for k in ('psm','rm','dm'))
        report['full_packet_bytes']=full['packet_bytes'];report['single_full_packet_equal_three_events']=True
        repeated=model(data)
        assert all(torch.equal(repeated[k],wire[k]) for k in ('psm','rm','dm'))
        report['repeated_forward_has_no_state_leak']=True
        model.packet_roundtrip=False;combined=model(data);errors=[]
        for i in range(2):
            one=train_utils.to_device(dataset.collate_batch_train([samples[i]]),torch.device('cuda'))['ego']
            def subset(sparse):
                active=sparse.indices[:,0]==i;indices=sparse.indices[active].clone();indices[:,0]=0
                return spconv.SparseConvTensor(sparse.features[active],indices,sparse.spatial_shape,1)
            small_ego,small_road=subset(ego),subset(road)
            model.encode_membrane=lambda lidar,n:(small_ego,membrane[i:i+1])
            model.encode_lidar=lambda lidar,n:small_road
            isolated=model(one)
            for key in ('psm','rm','dm'):
                errors.append(float((combined[key][i:i+1]-isolated[key]).abs().max()))
                assert torch.allclose(combined[key][i:i+1],isolated[key],atol=2e-5,rtol=2e-5)
        report['batch_isolation_max_response_error']=max(errors)
        if joint:
            # Restore full cached batch, then perturb only the local pre-count
            # state. Changes must reach actual detector outputs, not a dead hook.
            model.encode_membrane=lambda lidar,n:(ego,membrane)
            model.encode_lidar=lambda lidar,n:road
            original_output=model(data)
            model.encode_membrane=lambda lidar,n:(ego,membrane+.25)
            changed=model(data)
            delta=max(float((changed[k]-original_output[k]).abs().max()) for k in ('psm','rm','dm'))
            assert delta>0
            report['local_membrane_perturbation_response_max']=delta
            from opencood.models.fuse_modules.e3dsnn_progressive import count_readout
            same_count=count_readout(membrane)+.4
            assert torch.equal(count_readout(same_count),count_readout(membrane))
            model.encode_membrane=lambda lidar,n:(ego,same_count)
            altered=model(data)
            same_count_delta=max(float((altered[k]-original_output[k]).abs().max()) for k in ('psm','rm','dm'))
            if joint_variant=='no_local':assert same_count_delta==0
            report['same_count_membrane_perturbation_response_max']=same_count_delta
    model.encode_lidar=original;model.encode_membrane=original_membrane
    # Verify local path equals the source checkpoint, not just our own dropout.
    source_cfg=load_yaml(str(source_config));legacy=train_utils.create_model(source_cfg).cuda().eval()
    legacy.load_state_dict(torch.load(str(ckpt),map_location='cpu'),strict=True)
    with torch.no_grad():
        saved=model.fusion_mode;model.fusion_mode='ego';legacy.fusion_mode='ego'
        a=model(data);b=legacy(data)
        assert all(torch.equal(a[k],b[k]) for k in ('psm','rm','dm'));model.fusion_mode=saved
    del legacy;report['ego_matches_mature_source_exactly']=True
    model.train();model.message_dropout=0.
    enc,dec,bn=encoder_digest(model),decoder_digest(model),bn_digest(model)
    report.update(encoder_sha256=enc,decoder_sha256=dec,bn_sha256=bn)
    criterion=train_utils.create_loss(cfg).cuda();optimizer=train_utils.setup_optimizer(cfg,model)
    optimizer.zero_grad(set_to_none=True);output,loss=checked_forward(model,criterion,batch)
    assert len(output['prefix_outputs'])==3;loss.backward()
    grads={k:float(p.grad.abs().sum()) for k,p in model.named_parameters() if p.grad is not None}
    expected={k for k,p in model.named_parameters() if p.requires_grad}
    assert set(grads)==expected and all(np.isfinite(v) for v in grads.values())
    if joint:
        # A ranking head's common score bias can have zero gradient: adding a
        # constant cannot change ordering. Require each functional block instead.
        blocks=['content.project.','content.expand.','joint_receiver.']
        if joint_variant!='frozen_order':blocks.append('progressive_sender.')
        else:assert not any(k.startswith('progressive_sender.') for k in grads)
        for prefix in blocks:
            assert sum(v for k,v in grads.items() if k.startswith(prefix))>0
    else:assert all(v>0 for v in grads.values())
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
    assert encoder_digest(model)==enc and decoder_digest(model)==dec and bn_digest(model)==bn
    assert (model.content.basis_digest()!=report['initial_basis_sha256'])==learn
    report.update(gradient_l1=grads,frozen_encoder_decoder_bn=True,loss_components=model.last_loss_components)
    model.message_dropout=1.
    with torch.no_grad():
        dropped=model(data)
        assert all(torch.equal(p[k],dropped['ego_output'][k]) for p in dropped['prefix_outputs'] for k in ('psm','rm','dm'))
    report.update(full_dropout_all_prefixes_equal_ego=True,passed=True,finished_unix=time.time(),
        scope='Augmented train frames 0/1; audit update discarded; not an AP or mean-byte result')
    (root/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',default='/data1/tzm/workspace/diagnostics')
    p.add_argument('--family',choices=('snn','ann'),required=True);p.add_argument('--learn',action='store_true')
    p.add_argument('--joint',action='store_true')
    p.add_argument('--joint-variant',choices=('full','no_local','clear_interbatch','frozen_order'),default='full')
    a=p.parse_args();audit(a.root,a.base,a.family,a.learn,a.joint,a.joint_variant)
