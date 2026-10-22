"""Registered three-family joint study. Audit all groups before training.

Paths are explicit, no hard-coded host repo. Atomic state/checkpoints, bounded
phases, no automatic scientific retries or hyperparameter search.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from opencood.tools import train_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools.train_e3dsnn_single import checked_forward,validate,write_json,save_torch,seed_worker

REPO=Path(__file__).resolve().parents[2]
FAMILIES=('residual_snn','old_snn','residual_ann')
SHA={'snn':'3c1a2ae7bc23eab1318f805b960aed4194bae153f27b849f692f0bfeda1fdeb8',
     'ann':'d0dc7b9ab4ccaad531a8fc63e0a7e8b91a62ab5fa7cf4d0740e366c6422b4bd8'}
SEED=20260929


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)


def prepare(root,base,sources):
    root.mkdir(parents=True,exist_ok=False)
    raw=yaml.safe_load(base.read_text());provenance={}
    for family in FAMILIES:
        kind='ann' if family=='residual_ann' else 'snn';source=sources/(kind+'.pth')
        assert digest(source)==SHA[kind]
        seed();cfg=copy.deepcopy(raw)
        cfg['model']['core_method']='e3dsnn_residual_coop'
        cfg['model']['args'].update(activation='relu' if kind=='ann' else 'count4',receiver_activation='relu',
            freeze_detector=False,freeze_detector_bn=True,enable_ego_aux=True,ego_aux_weight=.5,
            residual_stream=dict(family=family,steps=16,event_cap=128,budget_bytes=4096,
                                 train_budget_range=[2048,8192],rate_weight=.001,checkpoint_identity=SHA[kind]))
        cfg['test_dir']=cfg['validate_dir'];cfg['train_params']=dict(batch_size=2,epoches=20)
        folder=root/family;folder.mkdir();p=folder/'config.yaml';p.write_text(yaml.safe_dump(cfg))
        model=train_utils.create_model(load_yaml(str(p)))
        donor=torch.load(str(source),map_location='cpu');state=model.state_dict()
        mapped=[k for k in state if k.startswith(('backbone_3d.','backbone_2d.','cls_head.','reg_head.','dir_head.'))]
        assert all(k in donor and donor[k].shape==state[k].shape for k in mapped)
        for k in mapped:state[k]=donor[k]
        model.load_state_dict(state,strict=True);save_torch(folder/'initial.pth',model.state_dict())
        provenance[family]=dict(source=str(source),source_sha256=SHA[kind],mapped=mapped,
            intentionally_unused_source_keys=sorted(set(donor)-set(mapped)),initial_sha256=digest(folder/'initial.pth'),
            parameters=sum(p.numel() for p in model.parameters()),
            joint_trainable={k:p.numel() for k,p in model.named_parameters() if p.requires_grad},config_sha256=digest(p))
    write_json(root/'protocol.json',dict(seed=SEED,families=FAMILIES,provenance=provenance,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),
        epochs=20,interface_epochs=2,batch_size=2,selection='mean full-validation AP70 at 2048/4096/8192; epochs0,2,4,...20',
        final_budgets=[0,2048,3072,4096,6144,8192,None],objective='Ldet(B)+0.5Ldet(0)+0.001Rproxy; no teacher',
        split_sha256={k:digest(cfg[k]) for k in ('root_dir','validate_dir')},data_manifest_sha256=digest(cfg['data_manifest']),
        exit='complete all three then assess; no automatic local sweeps or extra training'))


def load(root,family,checkpoint='initial.pth'):
    seed();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.2)
    folder=root/family;cfg=load_yaml(str(folder/'config.yaml'))
    model=train_utils.create_model(cfg).cuda();state=torch.load(str(folder/checkpoint),map_location='cpu')
    model.load_state_dict(state,strict=True);model.contract_identity=digest(folder/checkpoint)
    return folder,cfg,model


def frozen_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if k.startswith(('backbone_3d.conv_input.','backbone_3d.conv1.','backbone_3d.conv2.','backbone_3d.conv3.'))
            or k.endswith(('running_mean','running_var','num_batches_tracked'))}


def check_frozen(model,before):
    state=model.state_dict()
    assert all(torch.equal(v,state[k].cpu()) for k,v in before.items()),'Frozen state changed'


def audit(root,family):
    folder,cfg,model=load(root,family);start=time.time();report={};model.eval()
    val=build_dataset(cfg,train=False);train=build_dataset(cfg,train=True)
    report['wire']=[]
    # Same source tensors for direct/wire; two real augmented frames.
    batch=train_utils.to_device(train.collate_batch_train([train[0],train[1]]),torch.device('cuda'))
    data=batch['ego'];original=model.encode_membrane
    with torch.no_grad():
        cached={id(data[k]):original(data[k],2) for k in ('processed_lidar','road_lidar')}
        model.encode_membrane=lambda lidar,b:cached[id(lidar)]
        for budget in (0,2048,4096):
            model.budget=budget;model.packet_roundtrip=False;a=model(data)
            model.packet_roundtrip=True;b=model(data)
            assert all(torch.equal(a[k],b[k]) for k in ('psm','rm','dm'))
            assert a['packet_bytes']==b['packet_bytes'] and max(b['packet_bytes'])<=budget
            report['wire'].append(dict(budget=budget,bytes=b['packet_bytes'],exact=True,records=model.last_records))
            if budget==0:
                model.fusion_mode='ego';c=model(data);model.fusion_mode='residual'
                assert all(torch.equal(c[k],b[k]) for k in ('psm','rm','dm'))
        model.encode_membrane=original
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    frozen=frozen_state(model);model.packet_roundtrip=False;model.budget=4096
    model.bounds=[4096,4096];model.set_stage('joint');model.train()
    criterion=train_utils.create_loss(cfg).cuda();opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=5e-5)
    torch.cuda.reset_peak_memory_stats();tic=time.perf_counter();opt.zero_grad(set_to_none=True)
    output,loss=checked_forward(model,criterion,batch)
    # A rate penalty alone must not satisfy the communication gradient gate.
    task=criterion(output,batch['ego']['label_dict'])
    task_names=[k for k,p in model.named_parameters() if p.requires_grad and k.startswith(('sender.','receiver.','legacy_receiver.'))]
    task_grads=torch.autograd.grad(task,[dict(model.named_parameters())[k] for k in task_names],retain_graph=True,allow_unused=True)
    task_gradient={prefix:sum(float(g.abs().sum()) for k,g in zip(task_names,task_grads) if k.startswith(prefix) and g is not None)
                   for prefix in ('sender.project.','sender.decode.','legacy_receiver.' if family=='old_snn' else 'receiver.')}
    assert all(np.isfinite(v) and v>0 for v in task_gradient.values()),task_gradient
    loss.backward()
    groups=('backbone_3d.conv4.','backbone_3d.conv_out.','backbone_2d.','cls_head.','reg_head.','dir_head.','sender.project.','sender.decode.',
            'legacy_receiver.' if family=='old_snn' else 'receiver.')
    gradient={prefix:sum(float(p.grad.abs().sum()) for k,p in model.named_parameters() if k.startswith(prefix) and p.grad is not None) for prefix in groups}
    assert all(np.isfinite(v) and v>0 for v in gradient.values()),gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step();check_frozen(model,frozen)
    torch.cuda.synchronize();report.update(gradient=gradient,task_only_gradient=task_gradient,step_seconds=time.perf_counter()-tic,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),step_records=model.last_records)
    model.load_state_dict(initial,strict=True);model.eval();model.bounds=[2048,8192]
    model.set_stage('interface');assert all(not p.requires_grad for k,p in model.named_parameters() if k.startswith(('backbone_3d.','backbone_2d.','cls_head.','reg_head.','dir_head.')))
    report.update(passed=True,zero_equals_local=True,frozen_buffers_unchanged=True,optimizer_step_discarded=True,
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),
                  elapsed_seconds=time.time()-start,initial_sha256=digest(folder/'initial.pth'))
    target=folder/'audit_task_gradient.json' if (folder/'audit.json').exists() else folder/'audit.json'
    if target.exists():raise FileExistsError('Audit extension already recorded')
    write_json(target,report);print(json.dumps({k:report[k] for k in ('passed','task_only_gradient','step_seconds','peak_allocated_bytes')}),flush=True)


def evaluate_budgets(model,val,workers,budgets,output=None):
    model.eval();model.packet_roundtrip=True;results={}
    for budget in budgets:
        model.budget=budget;name='full' if budget is None else 'b'+str(budget)
        if output is None:results[name]=validate(model,val,workers)
        else:
            dest=output/name;dest.mkdir();packet_dir=dest/'packets';packet_dir.mkdir()
            records=[];original=model.forward
            def forward(data):
                result=original(data)
                # Preserve even empty packets, one frame per validation batch.
                packet=model.last_packets[0] if model.last_packets else b''
                path=packet_dir/('%06d.bin'%len(records));path.write_bytes(packet)
                record=model.last_records[0] if model.last_records else dict(frame_id=str(data['frame_ids'][0]),actual_bytes=0,events=0,steps=0)
                records.append(dict(record,file=path.name,sha256=digest(path)))
                return result
            model.forward=forward
            try:metrics=validate(model,val,workers,include_3d=True,prediction_dir=dest/'predictions',profile=True)
            finally:model.forward=original
            write_json(dest/'evaluation.json',metrics);write_json(packet_dir/'index.json',records)
            subprocess.run([sys.executable,'-m','opencood.tools.replay_e3dsnn_predictions','--evaluation-dir',str(dest)],cwd=str(REPO),check=True,timeout=3600)
            assert read(dest/'offline_replay.json')['passed']
            results[name]=dict(evaluation=metrics,replay=read(dest/'offline_replay.json'))
    return results


def train(root,family):
    protocol=read(root/'protocol.json')
    assert all(read(root/f/'audit.json')['passed'] for f in FAMILIES),'All three audits required'
    for family_gate in FAMILIES:
        path=root/family_gate/'audit_task_gradient.json'
        gate=read(path if path.exists() else root/family_gate/'audit.json')
        assert gate['passed'] and all(v>0 for v in gate['task_only_gradient'].values())
    folder,cfg,model=load(root,family);out=folder/'train';out.mkdir(exist_ok=False)
    train_data=build_dataset(cfg,train=True);val=build_dataset(cfg,train=False)
    criterion=train_utils.create_loss(cfg).cuda();frozen=frozen_state(model)
    generator=torch.Generator().manual_seed(SEED)
    loader=DataLoader(train_data,batch_size=2,shuffle=True,num_workers=2,collate_fn=train_data.collate_batch_train,
                      generator=generator,worker_init_fn=seed_worker,multiprocessing_context='spawn')
    # Include all joint parameters at construction; requires_grad controls warmup.
    model.set_stage('joint');opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=1e-4)
    best=-1.;history=[];started=time.time()
    for epoch in range(21):
        if time.time()-started>5*24*3600:raise TimeoutError('Five-day study group cap')
        if epoch:
            model.set_stage('interface' if epoch<=2 else 'joint');model.train();model.packet_roundtrip=False
            if epoch in (15,19):
                for group in opt.param_groups:group['lr']*=.1
            total=0.;samples=0;record_totals=dict(bytes=0,events=0,steps=0)
            for index,batch in enumerate(loader):
                batch=train_utils.to_device(batch,torch.device('cuda'));opt.zero_grad(set_to_none=True)
                output,loss=checked_forward(model,criterion,batch);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
                n=batch['ego']['object_bbx_mask'].shape[0];total+=float(loss.detach())*n;samples+=n
                for rec in model.last_records:
                    for key,field in (('bytes','actual_bytes'),('events','events'),('steps','steps')):record_totals[key]+=rec[field]
                if index%50==0:
                    write_json(folder/'status.json',dict(state='training',epoch=epoch,batch=index+1,total_batches=len(loader),loss=float(loss.detach()),
                        communication_means={k:v/samples for k,v in record_totals.items()}))
            assert samples==4678;check_frozen(model,frozen)
            save_torch(out/'last.pt',dict(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),
                random_state=random.getstate(),numpy_state=np.random.get_state(),torch_state=torch.get_rng_state(),
                cuda_state=torch.cuda.get_rng_state_all(),loader_state=generator.get_state(),protocol_sha256=digest(root/'protocol.json')))
        if epoch%2==0:
            write_json(folder/'status.json',dict(state='validating',epoch=epoch))
            model.contract_identity=digest(out/'last.pt') if epoch else digest(folder/'initial.pth')
            rows=evaluate_budgets(model,val,2,(2048,4096,8192))
            assert all(v['samples']==1738 and v['gt_boxes']==28622 for v in rows.values())
            score=sum(v['ap']['0.7'] for v in rows.values())/3
            entry=dict(epoch=epoch,mean_ap70=score,evaluations=rows)
            history.append(entry);write_json(out/'history.json',history)
            if score>best:
                best=score;save_torch(out/'best.pth',model.state_dict());write_json(out/'best_metrics.json',entry)
    model.load_state_dict(torch.load(str(out/'best.pth'),map_location='cpu'),strict=True)
    model.contract_identity=digest(out/'best.pth');check_frozen(model,frozen)
    final=folder/'final';final.mkdir();write_json(folder/'status.json',dict(state='final_evaluation'))
    result=evaluate_budgets(model,val,2,(0,2048,3072,4096,6144,8192,None),final)
    write_json(folder/'summary.json',dict(passed=True,best=read(out/'best_metrics.json'),evaluations=result,
        checkpoint_sha256=digest(out/'best.pth'),frozen_shallow_and_bn_verified=True))
    write_json(folder/'status.json',dict(state='complete',summary_sha256=digest(folder/'summary.json')))


def coordinate(root):
    from opencood.tools.launch_e3dsnn_joint_round import check_gpu
    for family in FAMILIES:
        gate=root/family/'audit_task_gradient.json'
        if not gate.exists():gate=root/family/'audit.json'
        audit_report=read(gate)
        assert audit_report['passed'] and all(v>0 for v in audit_report['task_only_gradient'].values())
    jobs=[]
    try:
        for gpu,family in enumerate(FAMILIES):
            folder=root/family
            if (folder/'train_launch.json').exists():raise FileExistsError('No duplicate training launch')
            admission=check_gpu(gpu,True,.2)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(REPO))
            with (folder/'train.log').open('xb') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.run_e3dsnn_residual_main','--root',str(root),
                    '--mode','train','--family',family],cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((family,proc,time.time()))
            write_json(folder/'train_launch.json',dict(pid=proc.pid,gpu=gpu,admission=admission,started_unix=time.time(),
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip()))
        while jobs:
            pending=[];states={}
            for family,proc,started in jobs:
                code=proc.poll()
                if code is None:
                    if time.time()-started>5*24*3600:
                        proc.terminate();raise TimeoutError('Own experiment timeout: '+family)
                    pending.append((family,proc,started));states[family]='running'
                elif code:
                    states[family]='failed';write_json(root/'group_failure.json',dict(family=family,exit_code=code))
                else:states[family]='complete'
            write_json(root/'pipeline_status.json',dict(state='running' if pending else 'finished',groups=states))
            jobs=pending
            if jobs:time.sleep(30)
        if (root/'group_failure.json').exists():raise RuntimeError('A main group failed; preserve other completed groups')
        results={f:read(root/f/'summary.json') for f in FAMILIES}
        write_json(root/'summary.json',dict(passed=True,results=results,protocol=read(root/'protocol.json')))
        write_json(root/'pipeline_status.json',dict(state='complete',summary_sha256=digest(root/'summary.json')))
    except Exception:
        # Do not kill other valid groups on one group's failure. Their own
        # status remains authoritative; no automatic replacement run.
        write_json(root/'pipeline_failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=('prepare','audit','train','coordinate'),required=True);p.add_argument('--family',choices=FAMILIES)
    p.add_argument('--base',type=Path);p.add_argument('--sources',type=Path);a=p.parse_args()
    try:
        if a.mode=='prepare':prepare(a.root,a.base,a.sources)
        elif a.mode=='audit':audit(a.root,a.family)
        elif a.mode=='coordinate':coordinate(a.root)
        else:train(a.root,a.family)
    except Exception:
        dest=a.root/a.family if a.family else a.root
        if dest.exists():write_json(dest/'failure.json',dict(error=traceback.format_exc(),mode=a.mode,no_automatic_retry=True))
        raise
