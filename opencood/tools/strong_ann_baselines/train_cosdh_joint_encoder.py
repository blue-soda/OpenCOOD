"""Three-epoch SNN adaptation with explicit trainable sets and fixed BN.

Optional ANN readout or bounded synaptic gains are separate registered studies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zlib


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','model-dir','data-root','train-split','val-split','sender-thresholds','readout-thresholds','front-thresholds','second-thresholds','third-thresholds','pillar-thresholds','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--receiver',choices=('original','geometry'),required=True)
    p.add_argument('--adaptation-scope',choices=('encoder_decoder','encoder_decoder_head'),default='encoder_decoder')
    p.add_argument('--synapse-mode',choices=('none','stage','channel'),default='none')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--train-quotas',type=int,nargs=3,default=None)
    p.add_argument('--spiking-io',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(a.repo).resolve()));os.chdir(a.repo)
    import numpy as np
    import torch
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from cosdh_counts import CoSDHCounts
    from cosdh_readout import CoSDHReadout
    from cosdh_membrane import CoSDHMembrane
    from cosdh_front_stage import CoSDHFrontStage
    from cosdh_pillar import CoSDHPillar
    from cosdh_geometry import GeometryBackend
    random.seed(303);np.random.seed(303);torch.manual_seed(303);torch.cuda.manual_seed_all(303)
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.6)
    train_ids=json.loads(Path(a.train_split).read_text());val_ids=json.loads(Path(a.val_split).read_text())
    assert len(train_ids)==len(set(train_ids))==4678 and len(val_ids)==1738 and not set(train_ids)&set(val_ids)
    model_dir=Path(a.model_dir);checkpoint=model_dir/'net_epoch3.pth'
    cfg=yaml_utils.load_yaml(str(model_dir/'config.yaml'))
    cfg.update(data_dir=a.data_root,root_dir=a.train_split,validate_dir=a.train_split,test_dir=a.train_split,comm_range=1e6);cfg['fusion']['core_method']='intermediatelate'
    model=train_utils.create_model(cfg)
    from cosdh_synapse import install as install_synapse
    if a.synapse_mode!='none':assert a.receiver=='geometry' and a.adaptation_scope=='encoder_decoder'
    install_synapse(model,a.synapse_mode,a.readout_thresholds)
    original=torch.load(str(checkpoint),map_location='cpu')
    model.load_state_dict(original,strict=True);model.cuda().eval()
    spiking_names=[]
    if a.spiking_io:
        from cosdh_spiking_io import install as install_spiking
        spiking_names=install_spiking(model,a.readout_thresholds)
    for v in model.parameters():v.requires_grad_(False)
    names=[]
    for name,module in model.named_modules():
        comp_conv=isinstance(module,torch.nn.Conv2d) and name.startswith('naive_compressor_list.') and ('.decoder.' in name or (a.spiking_io and '.encoder.synapse' in name))
        conv=isinstance(module,torch.nn.Conv2d) and (name.startswith('backbone.blocks.') or comp_conv)
        pfn=name=='pillar_vfe.pfn_layers.0.linear' and isinstance(module,torch.nn.Linear)
        if conv or pfn:
            for suffix,v in module.named_parameters(recurse=False):v.requires_grad_(True);names.append(name+'.'+suffix)
    expected={'pillar_vfe.pfn_layers.0.linear.weight'}
    expected|={'backbone.blocks.%d.%d.weight'%(stage,1+3*i) for stage,n in enumerate((4,6,9)) for i in range(n)}
    assert {n for n in names if not n.startswith('naive_compressor_list.')}==expected
    registered=[n for n in names if n.startswith('naive_compressor_list.')]
    assert len(registered)==12, registered
    assert len(names)==len(expected)+len(registered)
    encoder_decoder_names=names[:]
    head_names=[]
    if a.adaptation_scope=='encoder_decoder_head':
        assert a.receiver=='geometry', 'Registered head adaptation uses the geometry receiver'
        for name,module in model.named_modules():
            downstream=(name.startswith('backbone.deblocks.') or name.startswith('shrink_conv.') or name in ('cls_head','reg_head','dir_head'))
            if downstream and isinstance(module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
                for suffix,v in module.named_parameters(recurse=False):
                    key=name+'.'+suffix;assert key not in names
                    v.requires_grad_(True);names.append(key);head_names.append(key)
        assert head_names and any(n.startswith('backbone.deblocks.') for n in head_names)
        assert any(n.startswith('shrink_conv.') for n in head_names)
        for prefix in ('cls_head.','reg_head.'):
            assert any(n.startswith(prefix) for n in head_names)
        if model.use_dir:assert any(n.startswith('dir_head.') for n in head_names)
    assert all(not v.requires_grad for module in model.modules() if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) for v in module.parameters())
    synapse_names=[]
    if a.synapse_mode!='none':
        for name,v in model.snn_synapse.named_parameters():
            v.requires_grad_(True);synapse_names.append('snn_synapse.'+name)
        assert len(synapse_names)==3
        names+=synapse_names
    parameters=[v for v in model.parameters() if v.requires_grad]
    groups=parameters
    if synapse_names:
        groups=[dict(params=[v for n,v in model.named_parameters() if v.requires_grad and n not in synapse_names],lr=1e-5),
                dict(params=list(model.snn_synapse.parameters()),lr=1e-3)]
    optimizer=torch.optim.AdamW(groups,lr=1e-5,weight_decay=1e-4);criterion=train_utils.create_loss(cfg)
    paths=1 if a.receiver=='geometry' else 2
    counts=CoSDHCounts(model,a.sender_thresholds,8,surrogate=True,quotas=a.train_quotas)
    readout=CoSDHReadout(model,a.readout_thresholds,'identity' if a.spiking_io else 'count16',surrogate=True)
    membrane=CoSDHMembrane(model,a.readout_thresholds,'local',16)
    stages=[CoSDHFrontStage(model,getattr(a,k+'_thresholds'),'count32',surrogate=True,audit_gradients=a.smoke,initial_voltage=v0,encoder_paths=paths) for k,v0 in [('front',0.),('second',.5),('third',0.)]]
    pillar=CoSDHPillar(model,a.pillar_thresholds,'count32',.5,surrogate=True,encoder_paths=paths,audit_gradients=a.smoke)
    geometry=GeometryBackend(model) if a.receiver=='geometry' else None
    converters=[counts,readout,membrane]+stages+[pillar]+([geometry] if geometry else [])
    dataset=build_dataset(cfg,visualize=False,train=False);assert list(dataset.split_info)==train_ids
    contract=dict(epochs=3,batch_size=1,seed=303,lr=1e-5,weight_decay=1e-4,optimizer='AdamW',grad_clip=10.,receiver=a.receiver,
        trainable_names=names,trainable_parameters=sum(v.numel() for v in parameters),encoder_paths=paths,
        adaptation_scope=a.adaptation_scope,encoder_decoder_names=encoder_decoder_names,head_names=head_names,
        synapse_mode=a.synapse_mode,synapse_names=synapse_names,spiking_io=a.spiking_io,spiking_names=spiking_names,synapse_lr=1e-3 if synapse_names else None,
        total_parameters=sum(v.numel() for v in model.parameters()),
        scope='PFN linear weight,19 BEV convolution weights,12 compressor-decoder tensors; optional ANN downstream convolutions or registered bounded synaptic gains. All BN and latent compressor encoders frozen.',
        checkpoint_sha256=sha(checkpoint),train_split_sha256=sha(a.train_split),val_split_sha256=sha(a.val_split),
        thresholds_sha256={k:sha(getattr(a,k+'_thresholds')) for k in ('sender','readout','front','second','third','pillar')},source_sha256=sha(__file__),
        preprocessing='4678 commontrain with inference preprocessing, no random augmentation; same seeded epoch permutation',
        selection='Fixed last epoch3; no intermediate AP or validation selection',surrogate='Exact IF forward; unit gradient inside calibrated range, zero outside',
        no_wire=True,no_byte_statistics=True,all_BN_frozen=True,train_event_quotas=a.train_quotas)
    write(out/'contract.json',contract)
    def forbidden(*args,**kwargs):raise AssertionError('Training must not compress/decompress')
    zlib.compress=zlib.decompress=forbidden
    history=[];updates=0;start=time.time()
    for epoch in range(1,2 if a.smoke else 4):
        order=np.random.RandomState(303+epoch).permutation(len(dataset))
        if a.smoke:order=order[:2]
        order_hash=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest();loss_sum=0.
        for batch_index,index in enumerate(order,1):
            batch=dataset.collate_batch_test([dataset[int(index)]]);batch=train_utils.to_device(batch,torch.device('cuda:0'))
            assert batch['ego']['record_len'].tolist()==[2]
            for c in converters:c.begin()
            optimizer.zero_grad(set_to_none=True);output=model(batch['ego']);routes=[c.finish() for c in converters]
            loss=criterion(output,batch['ego']['label_dict']);assert torch.isfinite(loss);loss.backward()
            gradients=dict(BEV=[c.gradient_summary() for c in stages],PFN=pillar.gradient_summary()) if (a.smoke and not a.spiking_io) else None
            if a.smoke and synapse_names:
                gradients['synapse']=[dict(nonzero=int(torch.count_nonzero(v.grad)),elements=v.numel(),l1=float(v.grad.abs().sum())) for v in model.snn_synapse.parameters()]
            norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters)
            if a.smoke:
                dead=[n for n,v in model.named_parameters() if v.requires_grad and (v.grad is None or torch.count_nonzero(v.grad)==0)]
                assert not dead, dead
            optimizer.step();updates+=1;loss_sum+=float(loss.detach())
            if batch_index==1 or batch_index%25==0 or a.smoke:
                row=dict(state='training',epoch=epoch,batch=batch_index,batches=len(order),optimizer_steps=updates,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_seconds=time.time()-start,order_sha256=order_hash,
                    PFN=routes[6],BEV=routes[3:6],receiver=routes[7] if geometry else routes[2])
                if gradients:row['gradients']=gradients
                if synapse_names:row['synapse']=model.snn_synapse.statistics()
                write(out/'progress.json',row)
                with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            del output,loss,batch
        state=model.state_dict();param_names={n for n,_ in model.named_parameters()}
    changed=[k for k,v in state.items() if k in param_names and (k not in original or not torch.equal(v.detach().cpu(),original[k]))]
        assert changed and set(changed).issubset(names) and all(torch.isfinite(v).all() for v in parameters), sorted(set(changed)-set(names))
        if a.smoke:assert set(changed)==set(names),'All registered tensors must update in admission'
        history.append(dict(epoch=epoch,samples=len(order),loss=loss_sum/len(order),order_sha256=order_hash,changed_names=changed,frozen_tensors_unchanged=True))
        write(out/'history.json',history)
        if a.smoke:
            write(out/'smoke.json',dict(passed=True,optimizer_steps=updates,discarded=True,frozen_tensors_unchanged=True,all_registered_tensors_updated=True,registered_tensor_count=len(names),both_agents_all20_IF_sites_nonzero_gradient=True,no_compression=True,history=history,contract=contract));return
        assert len(order)==4678 and shutil.disk_usage(str(out)).free>1024**3
        tmp=out/'last.tmp';torch.save(dict(model=state,optimizer=optimizer.state_dict(),epoch=epoch,history=history,contract=contract,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())),str(tmp));os.replace(str(tmp),str(out/'last.pt'))
    for c in reversed(converters):c.close()
    dest=out/'final_model';dest.mkdir();shutil.copyfile(str(model_dir/'config.yaml'),str(dest/'config.yaml'));torch.save(model.state_dict(),str(dest/'net_epoch3.pth'))
    write(out/'completed.json',dict(epochs=3,optimizer_steps=updates,no_ap_selection=True,frozen_tensors_unchanged=True,checkpoint_sha256=sha(dest/'net_epoch3.pth'),config_sha256=sha(dest/'config.yaml'),history=history))
    write(out/'progress.json',dict(state='complete',epoch=3,batch=4678,optimizer_steps=updates))


if __name__=='__main__':main()
