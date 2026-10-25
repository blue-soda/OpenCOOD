"""Two real-batch gradient probes; no optimizer step or checkpoint writes.

This prepares low-rate adaptation, not an AP experiment. Uses the official
stochastic training latent and mask STE; freezes BN statistics explicitly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback


def sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(a):
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ['OMP_NUM_THREADS'] = '1'
    sys.path.insert(0, str(Path(a.repo).resolve()))
    import numpy as np
    import torch
    import opencood
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    assert Path(a.repo).resolve() in Path(opencood.__file__).resolve().parents
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.7)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(20260929);np.random.seed(20260929)
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    checkpoint = Path(a.model_dir)/'net_epoch_bestval_at7.pth'
    assert sha(checkpoint) == '7375da04507b4f543666e15aa02897e41e3c32e08bcb3f74b27ad49fb28c803c'
    source_split = Path(a.train_split)
    manifest_path = Path(a.data_manifest)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['protocol'] == 'paired_dair_fusion_v1'
    assert sha(source_split) == manifest['splits']['train']['source_sha256'] == '805f09b1bd0b80cde81aa3e01983afa21f7796712db7c534ed0335c7842438aa'
    assert sha(Path(a.data_root)/'cooperative/data_info.json') == manifest['cooperative_info_sha256']
    ids = manifest['splits']['train']['valid_ids']
    assert len(ids) == len(set(ids)) == manifest['splits']['train']['valid_count'] == 4678
    assert set(ids).issubset(json.loads(source_split.read_text()))
    split = Path(a.output)/'common_train.json'
    split.write_text(json.dumps(ids,indent=2))
    hypes = yaml_utils.load_yaml(str(Path(a.model_dir)/'config.yaml'))
    hypes.update(data_dir=a.data_root,root_dir=str(split),validate_dir=str(split),
                 test_dir=str(split),comm_range=1000000)
    model = train_utils.create_model(hypes)
    state = torch.load(str(checkpoint),map_location='cpu')
    expected = model.state_dict()
    missing = sorted(set(expected)-set(state));extra = sorted(set(state)-set(expected))
    assert missing == ['ib_communication.mu_net.1.scale','ib_communication.mu_net.1.shift',
        'ib_communication.std_net.0.bias','ib_communication.std_net.0.weight',
        'ib_communication.std_net.1.scale','ib_communication.std_net.1.shift']
    assert extra == ['ib_communication.sigma_net.0.bias','ib_communication.sigma_net.0.weight']
    for suffix in ('weight','bias'):
        state['ib_communication.std_net.0.'+suffix]=state.pop('ib_communication.sigma_net.0.'+suffix)
    identity_keys=[]
    for module in ('mu_net','std_net'):
        for suffix,value in (('scale',1.),('shift',0.)):
            key='ib_communication.'+module+'.1.'+suffix
            state[key]=torch.full_like(expected[key],value);identity_keys.append(key)
    model.load_state_dict(state,strict=True);del state,expected
    for name,p in model.named_parameters():
        if name in identity_keys:p.requires_grad_(False)
    model=model.cuda().train()
    for mod in model.modules():
        if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.eval()
    before={n:v.detach().cpu().clone() for n,v in model.named_buffers()}
    dataset=build_dataset(hypes,visualize=False,train=True)
    assert list(dataset.split_info)==ids
    samples=[dataset[i] for i in range(2)]
    batch=dataset.collate_batch_train(samples)
    batch=train_utils.to_device(batch,torch.device('cuda'))
    assert batch['ego']['record_len'].tolist()==[2,2]
    loss_fn=train_utils.create_loss(hypes)
    namespace=model.ib_communication.get_mask.__func__.__globals__
    original=namespace['topk_mask'];rows=[]
    required=['ib_communication.encoder','ib_communication.decoder',
              'ib_communication.mu_net','ib_communication.std_net',
              'ib_communication.adapt_layer_in','ib_communication.adapt_layer_out',
              'backbone','cls_head','reg_head','dir_head']
    for ratio in (.01,.1):
        calls=[]
        def topk(mask,ratio=ratio):
            # The official caller supplies ratio=.1. Override it explicitly.
            value=original(mask,ratio=selected)
            calls.append({'input_shape':list(mask.shape),'selected_ratio':selected,
                          'nonzero':int(torch.count_nonzero(value).item())})
            return value
        selected=ratio;namespace['topk_mask']=topk
        model.zero_grad(set_to_none=True)
        output=model(batch['ego'])
        loss=loss_fn(output,batch['ego']['label_dict'])
        assert torch.isfinite(loss).item()
        loss.backward()
        groups={}
        for prefix in required:
            ps=[p for n,p in model.named_parameters() if n.startswith(prefix+'.') and p.requires_grad]
            gs=[p.grad for p in ps if p.grad is not None]
            assert gs and all(torch.isfinite(g).all().item() for g in gs),prefix
            norm=float(sum(g.detach().double().square().sum().item() for g in gs)**.5)
            assert norm>0,prefix
            groups[prefix]={'norm':norm,'with_grad':len(gs),'parameters':len(ps)}
        assert model.ib_communication.training and calls
        rows.append({'ratio':ratio,'det_loss':float(loss.detach()),
            'kl_value':float(output['KL_loss'].detach()),'kl_in_audit_loss':False,
            'mask_calls':calls,'gradient_groups':groups})
        del output,loss
    namespace['topk_mask']=original
    assert all(torch.equal(before[n],v.detach().cpu()) for n,v in model.named_buffers())
    return {'passed':True,'seed':20260929,'train_ids':ids[:2],'split_samples':4678,
        'batch_size':2,'record_len':[2,2],'probes':rows,'optimizer_steps':0,
        'bn_buffers_unchanged':True,'identity_compatibility_frozen':identity_keys,
        'stochastic_latent_preserved':True,'checkpoint_sha256':sha(checkpoint),
        'train_split_sha256':sha(split),'raw_train_split_sha256':sha(source_split),
        'data_manifest_sha256':sha(manifest_path),'script_sha256':sha(Path(__file__)),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Real-batch gradient readiness only; not a training run, AP result or budget parity claim'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('repo','model-dir','data-root','train-split','data-manifest','output','gpu'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:result=run(a)
    except Exception:
        (out/'failure.json').write_text(json.dumps({'passed':False,'error':traceback.format_exc()},indent=2));raise
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
