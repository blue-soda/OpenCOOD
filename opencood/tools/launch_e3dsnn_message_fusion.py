"""Fusion activation x received-message support, fixed sender and decoder."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import yaml
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, free_gpu, evaluation
from opencood.tools.launch_e3dsnn_budget_round import make_jobs as budget_jobs
from opencood.tools.train_e3dsnn_single import write_json, checked_forward
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.utils.e3dsnn_budget import decoder_digest
from opencood.utils.e3dsnn_fusion_probe import bn_digest

GROUPS = ('ann_geometry', 'ann_received', 'snn_geometry', 'snn_received')


def source_gate(base):
    root = Path(base)/'e3dsnn_budget_round1_20261018'
    assert read(root/'round_status.json')['state'] == 'complete'
    final, zero, audit = [read(root/n) for n in ('adapt10_summary.json','zero_summary.json','audit.json')]
    assert final['passed'] and zero['passed'] and audit['passed']
    initial = root/'s_fusion_initial.pth'
    assert sha(initial) == audit['groups']['s_fusion']['initial_sha256']
    assert sha(root/'s_fusion.yaml') == audit['groups']['s_fusion']['config_sha256']['']
    return dict(config=str(root/'s_fusion.yaml'), checkpoint=str(initial),
                checkpoint_sha256=sha(initial), final_summary_sha256=sha(root/'adapt10_summary.json'),
                zero=zero['groups']['s_fusion']['on'], audit=audit['groups']['s_fusion'],
                control=final['groups']['s_fusion'])


def audit(root, base):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'audit.json').exists(): raise FileExistsError('Already audited')
    source = source_gate(base)
    report = dict(source_summary_sha256=source['final_summary_sha256'],groups={})
    for group in GROUPS:
        raw = yaml.safe_load(Path(source['config']).read_text())
        kind, support = group.split('_')
        raw['name'] = 'message_'+group
        raw['model']['core_method'] = 'e3dsnn_message_fusion'
        raw['model']['args'].update(fusion_activation_kind='relu' if kind=='ann' else 'count4',
                                   message_support=support, train_encoder=False)
        for ego in (False, True):
            cfg = json.loads(json.dumps(raw))
            if ego: cfg['model']['args']['fusion_mode'] = 'ego'
            with (root/(group+('_ego' if ego else '')+'.yaml')).open('x') as f: yaml.safe_dump(cfg,f)
        initial=root/(group+'_initial.pth');shutil.copyfile(source['checkpoint'],str(initial))
        assert sha(initial)==source['checkpoint_sha256']
        torch.manual_seed(20260929);np.random.seed(20260929);random.seed(20260929)
        cfg=load_yaml(str(root/(group+'.yaml')));data=build_dataset(cfg,train=True)
        batch=train_utils.to_device(data.collate_batch_train([data[0],data[1]]),torch.device('cuda'))
        model=train_utils.create_model(cfg).cuda().eval()
        model.load_state_dict(torch.load(str(initial),map_location='cpu'),strict=True)
        original=model.encode_lidar
        with torch.no_grad():
            cached={id(batch['ego'][k]):original(batch['ego'][k],2) for k in ('processed_lidar','road_lidar')}
            model.encode_lidar=lambda lidar,n:cached[id(lidar)]
            direct=model(batch['ego']);model.packet_roundtrip=True;wire=model(batch['ego'])
            assert all(torch.equal(direct[k],wire[k]) for k in ('psm','rm','dm'))
            native_cfg=load_yaml(source['config'])
            if kind=='snn':
                native_cfg['model']['core_method']='e3dsnn_receiver_split'
                native_cfg['model']['args'].update(fusion_activation_kind='count4',bev_activation='relu',message_codec=raw['model']['args']['budget_codec'])
            native=train_utils.create_model(native_cfg).cuda().eval()
            native.load_state_dict(torch.load(str(initial),map_location='cpu'),strict=True)
            native.encode_lidar=lambda lidar,n:cached[id(lidar)];native.packet_roundtrip=True
            reference=native(batch['ego'])
            assert reference['packet_bytes']==wire['packet_bytes']
            if support=='geometry':
                assert all(torch.equal(reference[k],wire[k]) for k in ('psm','rm','dm'))
            empty_identity=None
            if support=='received':
                model.packet_roundtrip=False
                key=id(batch['ego']['road_lidar']);road=cached[key]
                cached[key]=road.replace_feature(torch.zeros_like(road.features))
                empty=model(batch['ego']);ego=model(dict(batch['ego'],disable_road=True))
                assert all(torch.equal(empty[k],ego[k]) for k in ('psm','rm','dm'))
                cached[key]=road;empty_identity=True
                del empty,ego,road
            del native,reference,cached
        model.encode_lidar=original;model.packet_roundtrip=False;model.train()
        enc,dec,bn=encoder_digest(model),decoder_digest(model),bn_digest(model)
        assert enc==source['audit']['encoder_sha256'] and dec==source['audit']['decoder_sha256']
        before={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
        assert all(k.startswith('fusion_net.') or k=='fusion_scale' for k in before)
        criterion=train_utils.create_loss(cfg).cuda();optimizer=train_utils.setup_optimizer(cfg,model)
        optimizer.zero_grad(set_to_none=True);_,loss=checked_forward(model,criterion,batch);loss.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        norm=sum(float(g.abs().sum()) for g in grads);assert norm>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
        assert encoder_digest(model)==enc and decoder_digest(model)==dec and bn_digest(model)==bn
        assert any(not torch.equal(before[k],p) for k,p in model.named_parameters() if k.startswith('fusion_net.'))
        report['groups'][group]=dict(initial_sha256=sha(initial),encoder_sha256=enc,decoder_sha256=dec,
            config_sha256={s:sha(root/(group+s+'.yaml')) for s in ('','_ego')},gradient_l1=norm,
            packet_bytes=wire['packet_bytes'],direct_wire_exact=True,geometry_native_exact=support=='geometry',
            empty_received_equals_ego=empty_identity,frozen_encoder_decoder_bn=True,
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),discarded_step_loss=float(loss.detach()))
        del model,original,criterion,optimizer,loss,before,batch,data,direct,wire,grads
        gc.collect();torch.cuda.empty_cache()
    assert len({x['parameters'] for x in report['groups'].values()})==1
    assert len({x['trainable_parameters'] for x in report['groups'].values()})==1
    report['passed']=True
    with (root/'audit.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


def make_jobs(root,group,stage):
    name,jobs=budget_jobs(root,group,stage)
    for job in jobs:
        job['name']=job['name'].replace('budget_','message_')
        job['command']=[s.replace('budget_'+group+'_'+stage,'message_'+group+'_'+stage) for s in job['command']]
    return name.replace('budget_','message_'),jobs


def check_bytes(on, reference):
    a,b=on['communication_frames'],reference['communication_frames']
    assert [x['frame_id'] for x in a]==[x['frame_id'] for x in b]
    diffs=[abs(x['bytes']-y['bytes']) for x,y in zip(a,b)]
    # Same cached sender must be byte-exact (audit). Independent CUDA sparse
    # forwards can differ slightly; bound and disclose this rather than hide it.
    assert max(diffs)<=64 and sum(diffs)/len(diffs)<=.1,'Frozen-message drift'
    return dict(max_absolute_frame_bytes=max(diffs),mean_absolute_frame_bytes=sum(diffs)/len(diffs))


def summarize(root,base,stage):
    source=source_gate(base);audit_result=read(root/'audit.json');report=dict(stage=stage,groups={})
    for group in GROUPS:
        name='message_'+group+'_'+stage
        assert read(root/(name+'_pipeline.json'))['state']=='complete'
        on=evaluation(root/(name+'_on'));entry=dict(on=on,message_drift=check_bytes(on,source['zero']))
        assert on['manifest']['message_support']==group.split('_')[1]
        if stage=='zero':
            assert on['manifest']['checkpoint_sha256']==source['checkpoint_sha256']
            if group=='ann_geometry':
                diff={k:on['evaluation']['ap'][k]-source['zero']['evaluation']['ap'][k] for k in ('0.3','0.5','0.7')}
                assert max(abs(x) for x in diff.values())<=.0002,'ANN control zero gate failed'
                entry['reference_ap_differences']=diff
        else:
            folder=root/name
            for n in ('fixed_bn.json','frozen_decoder.json','frozen_encoder.json'):
                check=read(folder/n)
                assert check['passed'] and check['verified_through_epoch']==10 and check['initial_sha256']==check['current_sha256']
                entry[n]=check
            assert entry['frozen_encoder.json']['current_sha256']==source['audit']['encoder_sha256']
            assert entry['frozen_decoder.json']['current_sha256']==source['audit']['decoder_sha256']
            entry['checkpoint_sha256']=sha(folder/'best_ap70.pth')
            assert on['manifest']['checkpoint_sha256']==entry['checkpoint_sha256']
            cfg=load_yaml(str(root/(group+'.yaml')));model=train_utils.create_model(cfg)
            model.load_state_dict(torch.load(str(folder/'best_ap70.pth'),map_location='cpu'),strict=True)
            assert encoder_digest(model)==source['audit']['encoder_sha256'] and decoder_digest(model)==source['audit']['decoder_sha256']
            entry['best_metrics']=read(folder/'best_ap70_metrics.json');entry['off']=evaluation(root/(name+'_off'))
            assert entry['off']['manifest']['checkpoint_sha256']==entry['checkpoint_sha256']
            assert max(abs(entry['off']['evaluation']['ap'][k]-source['control']['off']['evaluation']['ap'][k]) for k in ('0.3','0.5','0.7'))<=.0002
            diag=read(root/(name+'_diagnostic.json'));assert diag['passed'];entry['gt_transitions70']=diag['gt_transitions70']
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
                    p=subprocess.Popen([sys.executable,str(root/'supervise.py'),str(plan)],cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
    source=source_gate(base);report=read(root/'audit.json');assert report['passed']
    assert report['source_summary_sha256']==source['final_summary_sha256']
    if len(gpus)!=4 or len(set(gpus))!=4:raise ValueError('Four distinct GPUs required')
    if (root/'launch.json').exists():raise FileExistsError('Already attempted')
    for g in GROUPS:
        assert sha(root/(g+'_initial.pth'))==source['checkpoint_sha256']
        for s in ('','_ego'):assert sha(root/(g+s+'.yaml'))==report['groups'][g]['config_sha256'][s]
    if shutil.disk_usage(str(root)).free<8*1024**3:raise RuntimeError('Insufficient reserve')
    for gpu in gpus:free_gpu(gpu)
    shutil.copyfile(str(Path(base)/'e3dsnn_budget_round1_20261018/supervise.py'),str(root/'supervise.py'))
    setup=dict(gpus=gpus,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),audit_sha256=sha(root/'audit.json'),started_unix=time.time())
    with (root/'launch.json').open('x') as f:json.dump(setup,f,indent=2)
    with (root/'coordinator.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.launch_e3dsnn_message_fusion','--root',str(root),'--base',str(base),'--coordinate'],cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
