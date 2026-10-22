"""Training-only label oracle: can task gradients identify harmful sent events?

Not a deployable sender, AP experiment, optimizer, or differentiable-byte claim.
The finite discrete deletions test the Count4 surrogate rather than trusting it.
"""
import argparse
from dataclasses import asdict
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
from opencood.tools.profile_e3dsnn_snapshot_charge import SOURCE,DIGEST
from opencood.tools.launch_e3dsnn_hybrid_round import sha
from opencood.tools.train_e3dsnn_single import write_json,checked_forward


def delete_events(counts,gradient,seed):
    c=np.asarray(counts);g=np.asarray(gradient)
    if c.shape!=g.shape or not np.isfinite(c).all() or not np.isfinite(g).all() or np.any(c<0) or np.any(c!=np.floor(c)):
        raise ValueError('Expected finite gradient and nonnegative integer counts')
    addresses=np.repeat(np.arange(c.size),c.ravel().astype(np.int64))
    harmful=addresses[g.ravel()[addresses]>0]
    k=min(32,len(harmful))
    ranked=harmful[np.lexsort((harmful,-g.ravel()[harmful]))][:k]
    random_order=np.random.RandomState(seed).permutation(len(addresses))[:k]
    result={}
    for name,chosen in (('oracle',ranked),('random',addresses[random_order])):
        removed=np.bincount(chosen,minlength=c.size).reshape(c.shape)
        new=c-removed
        assert np.all(new>=0) and int(removed.sum())==k
        result[name]=dict(counts=new,removed=removed,predicted_loss_delta=float(-(g*removed).sum()))
    return result,k


REPO=Path(__file__).resolve().parents[2]


def run(root,source=SOURCE):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.models.fuse_modules import e3dsnn_snapshot_adapter as adapter
    from opencood.utils import task_snapshot_packet as wire
    root.mkdir(parents=True,exist_ok=False)
    native=adapter.stream_readout
    try:
        seed=20260929;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.cuda.set_per_process_memory_fraction(.2)
        cfg=load_yaml(str(source/'train.yaml'));cfg['data_augment']=[]
        data=build_dataset(cfg,train=True)
        manifest=json.loads(Path(cfg['data_manifest']).read_text())
        assert set(data.data)==set(manifest['splits']['train']['valid_ids'])
        assert not set(data.data)&set(manifest['splits']['val']['valid_ids'])
        indices=np.linspace(0,len(data)-1,16,dtype=np.int64).tolist()
        assert len(set(indices))==16
        checkpoint=source/'train/best_ap70.pth';assert sha(checkpoint)==DIGEST
        model=train_utils.create_model(cfg).cuda().eval();state=torch.load(str(checkpoint),map_location='cpu')
        model.load_state_dict(state,strict=True);model.requires_grad_(False)
        model.snapshot_budget=2048;model.packet_roundtrip=True;model.snapshot_audit=True
        criterion=train_utils.create_loss(cfg).cuda();captured={}
        def observed(membrane,counts,pose,contract,receiver,decode):
            leaf=counts.detach().clone().requires_grad_(True)
            captured.update(membrane=membrane,counts=leaf,pose=pose,contract=contract,receiver=receiver,decode=decode)
            return native(membrane,leaf,pose,contract,receiver,decode)
        adapter.stream_readout=observed
        protocol=dict(checkpoint_sha256=DIGEST,source_config_sha256=sha(source/'train.yaml'),source=str(source),
            data_manifest_sha256=sha(Path(cfg['data_manifest'])),train_split_sha256=sha(Path(cfg['root_dir'])),
            split='train only; no validation samples',indices=indices,seed=seed,augmentation=[],budget_bytes=2048,
            deletion='up to 32 positive-gradient sent unit events; random same number, seed+frame index',
            frozen='all weights, BN, encoder, P/D, receiver, BEV and heads',
            loss='original fused classification, regression and direction training objective',
            gradients='Count4 surrogate and frozen support mask; finite discrete forward is authoritative',
            limitation='label oracle at receiver, not deployable sender or AP improvement; modified bytes measured, not assumed equal',
            gate='oracle mean loss lower than baseline and random, beats random in >=10/16 frames, changes baseline in >=8/16',
            exit='one 16-training-frame probe; no deletion-size/seed sweep; no optimizer or automatic architecture training',
            git_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip())
        write_json(root/'protocol.json',protocol);records=[];started=time.time()
        for index in indices:
            if time.time()-started>1800:raise TimeoutError('30 minute diagnostic limit')
            batch=train_utils.to_device(data.collate_batch_train([data[index]]),torch.device('cuda'))
            output,loss=checked_forward(model,criterion,batch)
            c=captured['counts'];gradient=torch.autograd.grad(loss,c)[0]
            assert torch.isfinite(gradient).all()
            counts=c.detach().cpu().numpy();g=gradient.detach().cpu().numpy()
            variants,k=delete_events(counts,g,seed+index)
            contract=captured['contract'];meta=dict(contract=asdict(contract),source_to_ego=captured['pose'][0].cpu().tolist(),family='if8',schema='task-count32-v1')
            scales=np.ones(32,dtype=np.float32);packet=wire.encode(counts,scales,meta)
            assert len(packet)==output['packet_bytes'][0]<=2048
            assert hashlib.sha256(packet).hexdigest()==model.snapshot_records[0]['packet_sha256']
            folder=root/str(index);folder.mkdir();(folder/'baseline.bin').write_bytes(packet)
            row=dict(index=index,frame_id=str(data.data[index]),removed_events=k,baseline_loss=float(loss.detach()),baseline_bytes=len(packet),
                sent_events=int(counts.sum()),positive_gradient_event_fraction=float(counts[g>0].sum()/max(1,counts.sum())),
                gradient_l1=float(np.abs(g).sum()),variants={})
            with torch.no_grad():
                for name,v in variants.items():
                    pkt=wire.encode(v['counts'],scales,meta);decoded,paid=wire.decode(pkt,meta)
                    assert np.array_equal(decoded,v['counts']) and np.all(paid==1)
                    q=native(captured['membrane'],c.new_tensor(decoded),captured['pose'],contract,captured['receiver'],captured['decode'])
                    prediction=model.detect_bev(q.flatten(1,2))
                    from opencood.utils.e3dsnn_aux_loss import detection_loss
                    current,_=detection_loss(prediction,batch['ego']['label_dict'],criterion,0.)
                    assert torch.isfinite(current)
                    (folder/(name+'.bin')).write_bytes(pkt)
                    row['variants'][name]=dict(loss=float(current),loss_delta=float(current)-row['baseline_loss'],
                        predicted_loss_delta=v['predicted_loss_delta'],bytes=len(pkt),within_budget=len(pkt)<=2048,
                        sha256=hashlib.sha256(pkt).hexdigest(),removed_last_count_channels=int(((counts>0)&(v['counts']==0)).sum()))
            np.savez_compressed(str(folder/'source_gradient.npz'),counts=counts,gradient=g,oracle_removed=variants['oracle']['removed'],random_removed=variants['random']['removed'])
            records.append(row);write_json(root/'status.json',dict(state='running',frames=len(records),total=16))
            del output,loss,gradient,c
        assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items()) and sha(checkpoint)==DIGEST
        means={name:float(np.mean([row['baseline_loss'] if name=='baseline' else row['variants'][name]['loss'] for row in records])) for name in ('baseline','oracle','random')}
        beats=sum(row['variants']['oracle']['loss']<row['variants']['random']['loss'] for row in records)
        changed=sum(row['variants']['oracle']['loss']!=row['baseline_loss'] for row in records)
        gate=means['oracle']<min(means['baseline'],means['random']) and beats>=10 and changed>=8
        write_json(root/'summary.json',dict(passed=True,training_frames=16,all_state_unchanged=True,means=means,
            oracle_beats_random_frames=beats,oracle_changes_baseline_frames=changed,hypothesis_gate=gate,
            all_modified_packets_within_budget=all(v['within_budget'] for row in records for v in row['variants'].values()),
            records=records,protocol=protocol))
        write_json(root/'status.json',dict(state='complete',summary_sha256=sha(root/'summary.json')))
    except Exception:
        write_json(root/'status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry'));raise
    finally:adapter.stream_readout=native


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default=str(SOURCE));p.add_argument('--launch',action='store_true');p.add_argument('--gpu',type=int,default=0);a=p.parse_args();root=Path(a.root)
    if a.launch:
        from opencood.tools.launch_e3dsnn_joint_round import check_gpu
        if root.exists():raise FileExistsError(str(root))
        admission=check_gpu(a.gpu,True,.2)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with Path(str(root)+'.log').open('xb') as f:
            proc=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.probe_e3dsnn_event_utility','--root',str(root),'--source',a.source],cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(Path(str(root)+'_launch.json'),dict(pid=proc.pid,gpu=a.gpu,admission=admission,started_unix=time.time()))
        print(json.dumps(dict(pid=proc.pid,root=str(root))))
    else:run(root,Path(a.source))
