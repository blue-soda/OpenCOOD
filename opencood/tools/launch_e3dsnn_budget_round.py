"""Matched sender/fusion adaptation under fixed sparse-message policies."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import yaml
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, free_gpu, evaluation, make_jobs as hybrid_jobs
from opencood.tools.launch_e3dsnn_rate_probe import sources
from opencood.tools.train_e3dsnn_single import write_json, checked_forward
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.utils.e3dsnn_budget import decoder_digest
from opencood.utils.e3dsnn_fusion_probe import bn_digest

GROUPS = ('s_fusion', 's_joint', 'a_fusion', 'a_joint')


def source_gate(base):
    base = Path(base)
    assert read(base/'e3dsnn_rate_diagnostics_20261018/rate_diagnostics_pipeline.json')['state'] == 'complete'
    for name in ('snn_compression','ann_compression','full_gap','lowrate_gap'):
        assert read(base/'e3dsnn_rate_diagnostics_20261018'/(name+'.json'))['passed']
    prior = sources(base)
    refs = {}
    for kind, family, folder, variant in [
        ('s','ann_receiver','e3dsnn_rate_probe_20261018','keep25'),
        ('a','ann_full','e3dsnn_lowrate_ann_20261018','q2keep50')]:
        root = base/folder
        assert read(root/(family+'_status.json'))['state'] == 'complete'
        summary = read(root/(family+'_summary.json')); assert summary['passed']
        assert summary['source']['checkpoint_sha256'] == prior[family]['checkpoint_sha256']
        refs[kind] = dict(source=prior[family], reference=summary['variants'][variant],
                          summary_sha256=sha(root/(family+'_summary.json')))
    return refs


def audit(root, base):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if (root/'audit.json').exists(): raise FileExistsError('Already audited')
    refs = source_gate(base)
    report = dict(groups={}, source_summary_sha256={k:v['summary_sha256'] for k,v in refs.items()})
    for group in GROUPS:
        kind = group[0]; source = refs[kind]['source']
        raw = yaml.safe_load(Path(source['config']).read_text())
        raw['name'] = 'budget_'+group
        raw['model']['core_method'] = 'e3dsnn_budget'
        options = dict(bits=3,keep_fraction=.25) if kind=='s' else dict(bits=2,keep_fraction=.5)
        raw['model']['args'].update(budget_codec=options, train_encoder=group.endswith('joint'))
        raw['optimizer']['lr'] = 5e-5
        for ego in (False,True):
            path = root/(group+('_ego' if ego else '')+'.yaml')
            cfg = json.loads(json.dumps(raw))
            if ego: cfg['model']['args']['fusion_mode']='ego'
            with path.open('x') as f: yaml.safe_dump(cfg,f)
        initial = root/(group+'_initial.pth')
        shutil.copyfile(source['checkpoint'],str(initial))
        assert sha(initial)==source['checkpoint_sha256']
        torch.manual_seed(20260929);np.random.seed(20260929)
        cfg = load_yaml(str(root/(group+'.yaml')))
        data = build_dataset(cfg,train=True)
        batch = train_utils.to_device(data.collate_batch_train([data[0],data[1]]),torch.device('cuda'))
        model = train_utils.create_model(cfg).cuda().eval()
        model.load_state_dict(torch.load(str(initial),map_location='cpu'),strict=True)
        original = model.encode_lidar
        with torch.no_grad():
            cached = {id(batch['ego'][k]):original(batch['ego'][k],2) for k in ('processed_lidar','road_lidar')}
            model.encode_lidar=lambda lidar,n:cached[id(lidar)]
            direct=model(batch['ego']);model.packet_roundtrip=True;wire=model(batch['ego'])
            assert all(torch.equal(direct[k],wire[k]) for k in ('psm','rm','dm'))
            native_cfg=load_yaml(source['config']);native_cfg['model']['args']['message_codec']=options
            native=train_utils.create_model(native_cfg).cuda().eval()
            native.load_state_dict(torch.load(str(initial),map_location='cpu'),strict=True)
            native.encode_lidar=lambda lidar,n:cached[id(lidar)];native.packet_roundtrip=True
            reference=native(batch['ego'])
            assert all(torch.equal(reference[k],wire[k]) for k in ('psm','rm','dm'))
            assert reference['packet_bytes']==wire['packet_bytes']
            del native,reference,cached
        model.encode_lidar=original;model.packet_roundtrip=False;model.train()
        enc,dec,bn=encoder_digest(model),decoder_digest(model),bn_digest(model)
        before={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
        criterion=train_utils.create_loss(cfg).cuda();optimizer=train_utils.setup_optimizer(cfg,model)
        optimizer.zero_grad(set_to_none=True);_,loss=checked_forward(model,criterion,batch);loss.backward()
        assert all(p.grad is None for k,p in model.named_parameters() if k.startswith(('backbone_2d.','cls_head.','reg_head.','dir_head.')))
        gradients={}
        for prefix in ('backbone_3d.','fusion_net.'):
            grads=[p.grad for k,p in model.named_parameters() if k.startswith(prefix) and p.grad is not None]
            assert all(torch.isfinite(g).all() for g in grads)
            gradients[prefix]=sum(float(g.abs().sum()) for g in grads)
        assert gradients['fusion_net.']>0
        assert (gradients['backbone_3d.']>0)==model.train_encoder
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
        assert decoder_digest(model)==dec and bn_digest(model)==bn
        assert (encoder_digest(model)!=enc)==model.train_encoder
        assert any(not torch.equal(before[k],p) for k,p in model.named_parameters() if k.startswith('fusion_net.'))
        report['groups'][group]=dict(initial_sha256=sha(initial),encoder_sha256=enc,decoder_sha256=dec,
            config_sha256={suffix:sha(root/(group+suffix+'.yaml')) for suffix in ('','_ego')},
            gradient_l1=gradients,packet_bytes=wire['packet_bytes'],direct_wire_native_exact=True,
            decoder_and_bn_fixed=True,encoder_changed=model.train_encoder,discarded_step_loss=float(loss.detach()))
        del model,original,criterion,optimizer,loss,before,batch,data,direct,wire
        gc.collect();torch.cuda.empty_cache()
    report['passed']=True
    with (root/'audit.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


def make_jobs(root,group,stage):
    name,jobs=hybrid_jobs(root,group,stage)
    for job in jobs:
        job['name']=job['name'].replace('hybrid_','budget_')
        cmd=[]
        for x in job['command']:
            if x.endswith('e3dsnn_hybrid_'+group+'_181.yaml'):x=str(root/(group+'.yaml'))
            elif x.endswith('e3dsnn_hybrid_'+group+'_ego_181.yaml'):x=str(root/(group+'_ego.yaml'))
            x=x.replace('hybrid_'+group+'_'+stage,'budget_'+group+'_'+stage)
            cmd.append(x)
        job['command']=cmd
    return name.replace('hybrid_','budget_'),jobs


def summarize(root,base,stage):
    refs=source_gate(base);audit_result=read(root/'audit.json');report=dict(stage=stage,groups={})
    for group in GROUPS:
        name='budget_'+group+'_'+stage
        assert read(root/(name+'_pipeline.json'))['state']=='complete'
        on=evaluation(root/(name+'_on'));entry=dict(on=on)
        if stage=='zero':
            assert on['manifest']['checkpoint_sha256']==audit_result['groups'][group]['initial_sha256']
            diff={k:on['evaluation']['ap'][k]-refs[group[0]]['reference']['evaluation']['ap'][k] for k in ('0.3','0.5','0.7')}
            assert max(abs(x) for x in diff.values())<=.0002,'Budget zero gate failed'
            entry['reference_ap_differences']=diff
        else:
            folder=root/name
            for n in ['fixed_bn.json','frozen_decoder.json']+(['frozen_encoder.json'] if group.endswith('fusion') else []):
                check=read(folder/n)
                assert check['passed'] and check['verified_through_epoch']==10 and check['initial_sha256']==check['current_sha256']
                entry[n]=check
            assert entry['frozen_decoder.json']['current_sha256']==audit_result['groups'][group]['decoder_sha256']
            entry['checkpoint_sha256']=sha(folder/'best_ap70.pth')
            assert on['manifest']['checkpoint_sha256']==entry['checkpoint_sha256']
            cfg=load_yaml(str(root/(group+'.yaml')));model=train_utils.create_model(cfg)
            model.load_state_dict(torch.load(str(folder/'best_ap70.pth'),map_location='cpu'),strict=True)
            entry['best_encoder_changed']=encoder_digest(model)!=audit_result['groups'][group]['encoder_sha256']
            assert entry['best_encoder_changed']==group.endswith('joint')
            assert decoder_digest(model)==audit_result['groups'][group]['decoder_sha256']
            entry['best_metrics']=read(folder/'best_ap70_metrics.json')
            entry['off']=evaluation(root/(name+'_off'))
            assert entry['off']['manifest']['checkpoint_sha256']==entry['checkpoint_sha256']
            diag=read(root/(name+'_diagnostic.json'));assert diag['passed']
            entry['gt_transitions70']=diag['gt_transitions70']
            del model
        report['groups'][group]=entry
    report['passed']=True
    with (root/(stage+'_summary.json')).open('x') as f:json.dump(report,f,indent=2)


def coordinate(root,base):
    setup=read(root/'launch.json')
    try:
        for stage in ('zero','adapt10'):
            if stage=='adapt10':assert read(root/'zero_summary.json')['passed']
            for gpu in setup['gpus']:free_gpu(gpu)
            processes=[]
            for group,gpu in zip(GROUPS,setup['gpus']):
                name,jobs=make_jobs(root,group,stage);plan=root/(name+'_plan.json')
                with plan.open('x') as f:json.dump(dict(name=name,jobs=jobs,gpu=gpu),f,indent=2)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMPY_MADVISE_HUGEPAGE='0')
                with (root/(name+'_supervisor.log')).open('xb') as log:
                    p=subprocess.Popen([sys.executable,str(root/'supervise.py'),str(plan)],cwd=str(REPO),env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append((name,p))
            write_json(root/'round_status.json',dict(state='running',stage=stage,supervisors={n:p.pid for n,p in processes}))
            while any(p.poll() is None for _,p in processes):
                if any(p.poll() not in (None,0) for _,p in processes):raise RuntimeError('Pipeline failed')
                time.sleep(15)
            if any(p.returncode for _,p in processes):raise RuntimeError('Pipeline failed')
            summarize(root,base,stage)
        write_json(root/'round_status.json',dict(state='complete'))
    except Exception:
        write_json(root/'round_status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry or termination of other pipelines.'))
        raise


def launch(root,base,gpus):
    root=Path(root);base=Path(base);refs=source_gate(base);report=read(root/'audit.json');assert report['passed']
    assert report['source_summary_sha256']=={k:v['summary_sha256'] for k,v in refs.items()}
    if len(gpus)!=4 or len(set(gpus))!=4:raise ValueError('Four distinct GPUs required')
    if (root/'launch.json').exists():raise FileExistsError('Already attempted')
    for g in GROUPS:
        assert sha(root/(g+'_initial.pth'))==report['groups'][g]['initial_sha256']
        for suffix in ('','_ego'):assert sha(root/(g+suffix+'.yaml'))==report['groups'][g]['config_sha256'][suffix]
    if shutil.disk_usage(str(root)).free<8*1024**3:raise RuntimeError('Insufficient reserve')
    for gpu in gpus:free_gpu(gpu)
    shutil.copyfile(str(base/'e3dsnn_bevscale_round1_20261018/supervise.py'),str(root/'supervise.py'))
    setup=dict(gpus=gpus,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),audit_sha256=sha(root/'audit.json'),started_unix=time.time())
    with (root/'launch.json').open('x') as f:json.dump(setup,f,indent=2)
    with (root/'coordinator.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.launch_e3dsnn_budget_round','--root',str(root),'--base',str(base),'--coordinate'],cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(coordinator_pid=p.pid,**setup)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--base',default='/data1/tzm/workspace/diagnostics')
    parser.add_argument('--gpus',type=int,nargs=4,default=[1,2,3,4])
    parser.add_argument('--audit',action='store_true');parser.add_argument('--coordinate',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    if args.audit:audit(root,args.base)
    elif args.coordinate:coordinate(root,args.base)
    else:launch(root,args.base,args.gpus)
