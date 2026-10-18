"""Bounded ANN response supervision: SNN/ANN students x KD=0/1."""
import argparse
import gc
import json
import math
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
from opencood.utils.e3dsnn_budget import decoder_digest
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.utils.e3dsnn_fusion_probe import bn_digest
from opencood.utils.e3dsnn_response_distill import load_teacher, state_digest, add_response_supervision
from opencood.utils.e3dsnn_aux_loss import detection_loss

GROUPS = ('s_kd0', 's_kd1', 'a_kd0', 'a_kd1')


def source_gate(base):
    base = Path(base); previous = base/'e3dsnn_budget_round1_20261018'
    assert read(previous/'round_status.json')['state']=='complete'
    summary=read(previous/'adapt10_summary.json');assert summary['passed']
    teacher=sources(base)['ann_full']
    assert teacher['checkpoint_sha256']=='624ce3b2adbd1eedc02f03876a33e38f5a5e66be7b6f9c8f78ae689cac558034'
    refs={}
    for kind in ('s','a'):
        group=kind+'_fusion';entry=summary['groups'][group]
        checkpoint=previous/('budget_'+group+'_adapt10')/'best_ap70.pth'
        assert sha(checkpoint)==entry['checkpoint_sha256']
        refs[kind]=dict(config=str(previous/(group+'.yaml')),checkpoint=str(checkpoint),
                       checkpoint_sha256=sha(checkpoint),reference=entry['on']['evaluation'])
    return dict(students=refs,teacher=teacher,parent_summary_sha256=sha(previous/'adapt10_summary.json'))


def audit(root,base):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError('Audit needs an empty directory')
    provenance=source_gate(base);report=dict(provenance=provenance,groups={})
    teacher_source=provenance['teacher']
    for group in GROUPS:
        source=provenance['students'][group[0]];weight=float(group[-1])
        raw=yaml.safe_load(Path(source['config']).read_text())
        raw['name']='response_'+group;raw['model']['args']['train_encoder']=True
        raw['response_distillation']=dict(weight=weight,config=teacher_source['config'],
            config_sha256=sha(teacher_source['config']),checkpoint=teacher_source['checkpoint'],
            checkpoint_sha256=teacher_source['checkpoint_sha256'])
        for suffix in ('','_ego'):
            cfg=json.loads(json.dumps(raw))
            if suffix:cfg['model']['args']['fusion_mode']='ego'
            with (root/(group+suffix+'.yaml')).open('x') as f:yaml.safe_dump(cfg,f)
        initial=root/(group+'_initial.pth');shutil.copyfile(source['checkpoint'],str(initial))
        assert sha(initial)==source['checkpoint_sha256']
        torch.manual_seed(20260929);np.random.seed(20260929)
        cfg=load_yaml(str(root/(group+'.yaml')))
        data=build_dataset(cfg,train=True)
        batch=train_utils.to_device(data.collate_batch_train([data[0],data[1]]),torch.device('cuda'))
        model=train_utils.create_model(cfg).cuda().eval()
        model.load_state_dict(torch.load(str(initial),map_location='cpu'),strict=True)
        teacher,teacher_info=load_teacher(cfg,torch.device('cuda'))
        teacher_hash=state_digest(teacher);dec=decoder_digest(model);bn=bn_digest(model);enc=encoder_digest(model)
        original=model.encode_lidar
        with torch.no_grad():
            cached={id(batch['ego'][k]):original(batch['ego'][k],2) for k in ('processed_lidar','road_lidar')}
            model.encode_lidar=lambda x,n:cached[id(x)]
            direct=model(batch['ego']);model.packet_roundtrip=True;wire=model(batch['ego'])
            assert all(torch.equal(direct[k],wire[k]) for k in ('psm','rm','dm'))
        model.encode_lidar=original;model.packet_roundtrip=False
        del cached,direct
        criterion=train_utils.create_loss(cfg).cuda();optimizer=train_utils.setup_optimizer(cfg,model)
        teacher_ids={id(p) for p in teacher.parameters()}
        assert not teacher_ids.intersection(id(p) for g in optimizer.param_groups for p in g['params'])
        model.train();model.message_dropout=0.
        output=model(batch['ego'])
        old,parts=detection_loss(output,batch['ego']['label_dict'],criterion,model.ego_aux_weight)
        zero=add_response_supervision(old,parts,output,batch['ego']['label_dict'],teacher,batch['ego'],0.)
        assert zero is old and torch.equal(output['message_kept'],torch.ones(2,device='cuda'))
        leaves=[output[k] for k in ('psm','rm','dm')]
        old_grad=torch.autograd.grad(old,leaves,retain_graph=True)
        zero_grad=torch.autograd.grad(zero,leaves,retain_graph=True)
        assert all(torch.equal(a,b) for a,b in zip(old_grad,zero_grad))
        del old,zero,output,old_grad,zero_grad,leaves
        # All dropped frames must add exactly zero KD, with the actual forward mask.
        model.message_dropout=1.
        with torch.no_grad():
            dropped,drop_loss=checked_forward(model,criterion,batch,teacher,1.)
            assert torch.equal(dropped['message_kept'],torch.zeros(2,device='cuda'))
            assert model.last_loss_components['kd_loss']==0.
        del dropped,drop_loss
        model.message_dropout=.1
        torch.cuda.synchronize();started=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        output,loss=checked_forward(model,criterion,batch,teacher,weight);loss.backward()
        gradients={}
        for prefix in ('backbone_3d.','fusion_net.'):
            grads=[p.grad for k,p in model.named_parameters() if k.startswith(prefix) and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            gradients[prefix]=sum(float(g.abs().sum()) for g in grads)
            assert gradients[prefix]>0
        assert all(p.grad is None for k,p in model.named_parameters() if k.startswith(('backbone_2d.','cls_head.','reg_head.','dir_head.')))
        assert all(p.grad is None for p in teacher.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
        torch.cuda.synchronize();seconds=time.monotonic()-started
        assert state_digest(teacher)==teacher_hash and decoder_digest(model)==dec and bn_digest(model)==bn
        assert encoder_digest(model)!=enc
        assert not any(k.startswith('teacher') for k in model.state_dict())
        report['groups'][group]=dict(initial_sha256=sha(initial),encoder_sha256=enc,decoder_sha256=dec,
            teacher=teacher_info,config_sha256={s:sha(root/(group+s+'.yaml')) for s in ('','_ego')},
            gradient_l1=gradients,packet_bytes=wire['packet_bytes'],step_seconds=seconds,
            kd0_old_loss_and_output_gradients_exact=True,teacher_fixed_and_isolated=True,
            actual_dropout_kd_zero=True,direct_wire_exact=True,loss_components=model.last_loss_components)
        del model,teacher,criterion,optimizer,batch,data,loss,output,wire,original,grads
        gc.collect();torch.cuda.empty_cache()
    # Conservative measured step extrapolation incl. 10 full validations and epoch zero.
    estimate=max(v['step_seconds'] for v in report['groups'].values())*2339*10+11*240
    report['estimated_train_seconds']=estimate
    report['train_timeout_seconds']=max(21600,int(math.ceil(estimate*2)))
    if report['train_timeout_seconds']>43200:raise RuntimeError('Audit predicts >12h bound; inspect before launch')
    report['passed']=True
    with (root/'audit.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


def make_jobs(root,group,stage):
    name,jobs=hybrid_jobs(root,group,stage)
    for job in jobs:
        job['name']=job['name'].replace('hybrid_','response_')
        command=[]
        for x in job['command']:
            if x.endswith('e3dsnn_hybrid_'+group+'_181.yaml'):x=str(root/(group+'.yaml'))
            elif x.endswith('e3dsnn_hybrid_'+group+'_ego_181.yaml'):x=str(root/(group+'_ego.yaml'))
            command.append(x.replace('hybrid_'+group+'_'+stage,'response_'+group+'_'+stage))
        if job['name'].endswith('_train'):
            command+=['--retain-initial-best']
            job['timeout_seconds']=read(root/'audit.json')['train_timeout_seconds']
        job['command']=command
    return name.replace('hybrid_','response_'),jobs


def summarize(root,base,stage):
    audit_result=read(root/'audit.json');refs=source_gate(base);report=dict(stage=stage,groups={})
    assert refs==audit_result['provenance']
    for group in GROUPS:
        name='response_'+group+'_'+stage
        assert read(root/(name+'_pipeline.json'))['state']=='complete'
        on=evaluation(root/(name+'_on'));entry=dict(on=on)
        audit_group=audit_result['groups'][group]
        if stage=='zero':
            assert on['manifest']['checkpoint_sha256']==audit_group['initial_sha256']
            diff={k:on['evaluation']['ap'][k]-refs['students'][group[0]]['reference']['ap'][k] for k in ('0.3','0.5','0.7')}
            assert max(abs(v) for v in diff.values())<=.0002,'Zero gate regression'
            entry['reference_ap_differences']=diff
        else:
            folder=root/name
            for filename in ('fixed_bn.json','frozen_decoder.json','frozen_teacher.json'):
                check=read(folder/filename)
                assert check['passed'] and check['verified_through_epoch']==10
                assert check['initial_sha256']==check['current_sha256'];entry[filename]=check
            assert entry['frozen_decoder.json']['current_sha256']==audit_group['decoder_sha256']
            assert entry['frozen_teacher.json']['current_sha256']==audit_group['teacher']['state_sha256']
            entry['checkpoint_sha256']=sha(folder/'best_ap70.pth')
            assert on['manifest']['checkpoint_sha256']==entry['checkpoint_sha256']
            entry['best_metrics']=read(folder/'best_ap70_metrics.json')
            entry['initial_metrics']=read(folder/'initial_metrics.json')
            assert entry['best_metrics']['ap']['0.7']>=entry['initial_metrics']['ap']['0.7']
            cfg=load_yaml(str(root/(group+'.yaml')));model=train_utils.create_model(cfg)
            model.load_state_dict(torch.load(str(folder/'best_ap70.pth'),map_location='cpu'),strict=True)
            assert decoder_digest(model)==audit_group['decoder_sha256']
            del model
            entry['off']=evaluation(root/(name+'_off'))
            assert entry['off']['manifest']['checkpoint_sha256']==entry['checkpoint_sha256']
            diag=read(root/(name+'_diagnostic.json'));assert diag['passed']
            entry['gt_transitions70']=diag['gt_transitions70']
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
        write_json(root/'round_status.json',dict(state='complete',finished_unix=time.time()))
    except Exception:
        write_json(root/'round_status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry or termination of other pipelines.'))
        raise


def launch(root,base,gpus):
    root=Path(root);base=Path(base);report=read(root/'audit.json');assert report['passed']
    assert report['provenance']==source_gate(base)
    if len(gpus)!=4 or len(set(gpus))!=4:raise ValueError('Four distinct GPUs required')
    if (root/'launch.json').exists():raise FileExistsError('Already attempted')
    for group in GROUPS:
        assert sha(root/(group+'_initial.pth'))==report['groups'][group]['initial_sha256']
        for suffix in ('','_ego'):
            assert sha(root/(group+suffix+'.yaml'))==report['groups'][group]['config_sha256'][suffix]
    if shutil.disk_usage(str(root)).free<8*1024**3:raise RuntimeError('Insufficient disk reserve')
    for gpu in gpus:free_gpu(gpu)
    shutil.copyfile(str(base/'e3dsnn_bevscale_round1_20261018/supervise.py'),str(root/'supervise.py'))
    setup=dict(gpus=gpus,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),audit_sha256=sha(root/'audit.json'),started_unix=time.time())
    with (root/'launch.json').open('x') as f:json.dump(setup,f,indent=2)
    with (root/'coordinator.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.launch_e3dsnn_distill','--root',str(root),'--base',str(base),'--coordinate'],cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
