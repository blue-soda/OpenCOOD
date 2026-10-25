"""Bounded low-rate InfoCom adaptation with resumable epoch checkpoints.

--smoke performs two disposable real optimizer steps and no weight export.
Formal runs require 6 GiB free at admission and space before each atomic write.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import audit_infocom_lowrate_training as readiness

GIB=1024**3
CONTRACT={'epochs':10,'batch_size':2,'seed':20260929,'lr':5e-5,
          'weight_decay':1e-4,'milestones':[7,9],'lr_gamma':.3,
          'mask_ratio':.01,'kl_beta':1e-4,'kl_start_adaptation_epoch':1,
          'train_scope':'all published weights; compatibility identity scales frozen',
          'bn':'all running buffers frozen','latent':'official train stochastic; eval mean',
          'selection':'highest common-GT AP70 at 1pct, epochs 1..10; earliest tie',
          'final_ratios':[.01,.025,.05,.1]}


def write_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(str(tmp),str(path))


def require_space(path,minimum):
    free=shutil.disk_usage(str(path)).free
    if free<minimum:raise RuntimeError('Insufficient checkpoint space: free=%d required=%d'%(free,minimum))
    return free


def save_atomic(torch,obj,path,estimate):
    require_space(path.parent,int(estimate*1.15)+512*1024**2)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,str(tmp));os.replace(str(tmp),str(path))


def rng_state(torch,np):
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(torch,np,state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])


def execute(a):
    root=Path(a.output)
    if not a.smoke:require_space(root,6*GIB)
    snapshot=root/'snapshot'
    files=('train_infocom_lowrate.py','audit_infocom_lowrate_training.py','evaluate_infocom.py',
           'replay_common_gt.py','infocom_wire.py','run_infocom_ratio_curve.py')
    if not a.resume:
        snapshot.mkdir()
        for name in files:shutil.copyfile(str(Path(__file__).resolve().parent/name),str(snapshot/name))
    snapshot_hashes={name:readiness.sha(snapshot/name) for name in files}
    admission=[]
    for unused in range(5):
        free,util=map(int,subprocess.check_output(['nvidia-smi','-i',a.gpu,
            '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().split(','))
        if free<18000 or util>20:raise RuntimeError('GPU admission failed')
        admission.append([free,util]);time.sleep(1)
    write_json(root/('admission_%d.json'%time.time_ns()),admission)
    model,dataset,hypes,buffers,identity,ids,checkpoint,split,raw_split,manifest=readiness.prepare(a)
    import numpy as np
    import torch
    from opencood.tools import train_utils
    from torch.utils.data import DataLoader,Subset
    assert json.loads((Path(a.evidence_root)/'common_val.json').read_text())==json.loads(manifest.read_text())['splits']['val']['valid_ids']
    namespace=model.ib_communication.get_mask.__func__.__globals__;original=namespace['topk_mask']
    def topk(mask,ratio=.1):return original(mask,ratio=CONTRACT['mask_ratio'])
    namespace['topk_mask']=topk
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=CONTRACT['lr'],weight_decay=CONTRACT['weight_decay'],eps=1e-10)
    criterion=train_utils.create_loss(hypes)
    estimated_model=sum(t.numel()*t.element_size() for t in model.state_dict().values())
    trainable_bytes=sum(p.numel()*p.element_size() for p in parameters)
    contract=dict(CONTRACT,checkpoint_sha256=readiness.sha(checkpoint),
        common_train_sha256=readiness.sha(split),manifest_sha256=readiness.sha(manifest),
        common_val_sha256=readiness.sha(Path(a.evidence_root)/'common_val.json'),
        script_sha256=readiness.sha(Path(__file__)),readiness_sha256=readiness.sha(Path(readiness.__file__)),
        model_bytes=estimated_model,trainable_bytes=trainable_bytes,snapshot_sha256=snapshot_hashes)
    best=None;best_weights=None;history=[];start=1
    if a.resume:
        state=torch.load(str(root/'last.pt'),map_location='cpu')
        assert state['contract']==contract
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        best=state['best'];best_weights=state['best_weights'];history=state['history'];start=state['epoch']+1
        restore_rng(torch,np,state['rng']);del state
    else:write_json(root/'protocol.json',contract)
    loader=DataLoader(Subset(dataset,[0,1,2,3]) if a.smoke else dataset,
        batch_size=2,shuffle=True,num_workers=0,drop_last=False,collate_fn=dataset.collate_batch_train)
    rows=[]
    def validate(epoch,ratio=.01,use_best=False):
        # A separate interpreter keeps official inference and our CPU evaluator
        # import namespaces isolated. Parent training RNG is untouched.
        candidate=root/('best' if use_best else 'candidate');candidate.mkdir(exist_ok=True)
        if not use_best:
            shutil.copyfile(str(Path(a.model_dir)/'config.yaml'),str(candidate/'config.yaml'))
            save_atomic(torch,model.state_dict(),candidate/'net_epoch0.pth',estimated_model)
        prefix='final_ratio%s'%ratio if use_best else 'val_epoch%02d'%epoch
        out=root/(prefix+'_attempt%d'%time.time_ns())
        source=snapshot
        assert all(readiness.sha(source/n)==h for n,h in snapshot_hashes.items())
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=a.our_repo)
        commands=[
            [sys.executable,str(source/'evaluate_infocom.py'),'--repo',a.repo,'--model-dir',str(candidate),
             '--data-root',a.data_root,'--split',str(Path(a.evidence_root)/'common_val.json'),
             '--output',str(out),'--limit','0','--gpu',a.gpu,'--comm-range','1000000','--mask-ratio',str(ratio),'--wire-codec',
             '--adaptation-protocol',str(root/'protocol.json')],
            [sys.executable,str(source/'replay_common_gt.py'),'--run',str(out),'--root',a.evidence_root]]
        for i,cmd in enumerate(commands):
            with (root/(out.name+'_%d.log'%i)).open('x') as log:
                subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400 if i==0 else 3600)
        sys.path.insert(0,str(snapshot))
        import run_infocom_ratio_curve as audit
        import infocom_wire as codec
        audit.ROOT=Path(a.evidence_root);audit.verify_wire(out,codec,1738)
        r=json.loads((out/'common_gt_replay.json').read_text())
        assert r['samples']==1738 and r['full_split']
        return r['common_gt_bev_ap']['0.7'],out
    for epoch in range(start,2 if a.smoke else 11):
        model.train()
        for mod in model.modules():
            if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.eval()
        lr=CONTRACT['lr']*CONTRACT['lr_gamma']**sum(epoch>m for m in CONTRACT['milestones'])
        for group in optimizer.param_groups:group['lr']=lr
        count=0;loss_sum=0.
        for step,batch in enumerate(loader,1):
            assert batch is not None
            batch=train_utils.to_device(batch,torch.device('cuda'))
            assert batch['ego']['record_len'].tolist()==[2,2]
            optimizer.zero_grad(set_to_none=True)
            output=model(batch['ego']);task=criterion(output,batch['ego']['label_dict'])
            kl=output['KL_loss'];loss=task+CONTRACT['kl_beta']*kl
            assert torch.isfinite(loss).item()
            loss.backward()
            # Detect nonfinite gradients, with the same clipping threshold as
            # our adaptation trainer; do not silently skip failed batches.
            norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
            optimizer.step();count+=2;loss_sum+=float(loss.detach())*2
            if step==1 or step%50==0 or a.smoke:
                row=dict(epoch=epoch,step=step,samples=count,loss=float(loss.detach()),
                    task_loss=float(task.detach()),kl=float(kl.detach()),gradient_norm=float(norm),lr=lr)
                write_json(root/'progress.json',row)
                if a.smoke:rows.append(row)
            del output,task,kl,loss
        assert count==(4 if a.smoke else 4678)
        assert all(torch.equal(buffers[n],v.detach().cpu()) for n,v in model.named_buffers())
        if a.smoke:
            assert len(rows)==2 and len(optimizer.state)>0
            write_json(root/'smoke.json',dict(passed=True,optimizer_steps=2,weights_discarded=True,
                bn_buffers_unchanged=True,rows=rows,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),contract=contract))
            return
        ap,out=validate(epoch)
        if best is None or ap>best['ap70']:
            best=dict(epoch=epoch,ap70=ap,validation=str(out))
            best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(dict(epoch=epoch,ap70=ap,loss=loss_sum/count,best=best))
        save_atomic(torch,dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=rng_state(torch,np),contract=contract,best=best,best_weights=best_weights,history=history),
            root/'last.pt',2*estimated_model+2*trainable_bytes)
        write_json(root/'history.json',history)
    dest=root/'best';dest.mkdir(exist_ok=True)
    shutil.copyfile(str(Path(a.model_dir)/'config.yaml'),str(dest/'config.yaml'))
    save_atomic(torch,best_weights,dest/'net_epoch0.pth',estimated_model)
    write_json(dest/'selection.json',best)
    write_json(root/'status.json',dict(state='trained',epochs=10,best=best,final_four_ratio_evaluation_required=True))
    final=[]
    for ratio in CONTRACT['final_ratios']:
        ap,out=validate(10,ratio,use_best=True)
        final.append(dict(ratio=ratio,ap70=ap,output=str(out),
            wire=json.loads((out/'summary.json').read_text())['wire_statistics']))
        write_json(root/'final_results_so_far.json',final)
    write_json(root/'status.json',dict(state='complete',epochs=10,best=best,final_results=final))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('repo','our-repo','model-dir','data-root','train-split','data-manifest','evidence-root','output','gpu'):p.add_argument('--'+n,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.smoke and a.resume:p.error('Smoke cannot resume')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=a.resume)
    try:execute(a)
    except Exception:
        write_json(root/'failure.json',dict(state='failed',error=traceback.format_exc(),automatic_retry=False));raise
