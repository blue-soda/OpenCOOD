"""Source-only utility prediction with held-out training frames and real bytes."""
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
from opencood.tools.profile_e3dsnn_snapshot_charge import DIGEST
from opencood.tools.launch_e3dsnn_hybrid_round import sha
from opencood.tools.train_e3dsnn_single import write_json,checked_forward
from opencood.tools.probe_e3dsnn_event_utility import delete_events
from opencood.models.fuse_modules.e3dsnn_source_utility import fit_probe,source_scores,select_suppressed

REPO=Path(__file__).resolve().parents[2]


def run(root,source):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.models.fuse_modules import e3dsnn_snapshot_adapter as adapter
    from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire
    from opencood.utils import task_snapshot_packet as wire
    from opencood.utils.e3dsnn_aux_loss import detection_loss
    root.mkdir(parents=True,exist_ok=False);native=adapter.stream_readout;hook=None
    try:
        seed=20260929;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.cuda.set_per_process_memory_fraction(.2)
        cfg=load_yaml(str(source/'train.yaml'));cfg['data_augment']=[]
        data=build_dataset(cfg,train=True);manifest=json.loads(Path(cfg['data_manifest']).read_text())
        assert set(data.data)==set(manifest['splits']['train']['valid_ids']) and not set(data.data)&set(manifest['splits']['val']['valid_ids'])
        indices=np.linspace(0,len(data)-1,128,dtype=np.int64).tolist();assert len(set(indices))==128
        checkpoint=source/'train/best_ap70.pth';assert sha(checkpoint)==DIGEST
        model=train_utils.create_model(cfg).cuda().eval();state=torch.load(str(checkpoint),map_location='cpu')
        model.load_state_dict(state,strict=True);model.requires_grad_(False)
        model.snapshot_budget=2048;model.packet_roundtrip=True;model.snapshot_audit=True
        criterion=train_utils.create_loss(cfg).cuda();captured={}
        def observed(membrane,counts,pose,contract,receiver,decode):
            leaf=counts.detach().clone().requires_grad_(True)
            captured.update(membrane=membrane,counts=leaf,pose=pose,contract=contract,receiver=receiver,decode=decode)
            return native(membrane,leaf,pose,contract,receiver,decode)
        def project_observed(module,inputs,output):
            captured.update(source_features=inputs[0].detach(),drive=torch.relu(output.detach()).clamp(max=8))
        adapter.stream_readout=observed;hook=model.content.project.register_forward_hook(project_observed)
        protocol=dict(checkpoint_sha256=DIGEST,source_config_sha256=sha(source/'train.yaml'),source=str(source),seed=seed,
            train_split_sha256=sha(Path(cfg['root_dir'])),data_manifest_sha256=sha(Path(cfg['data_manifest'])),
            fitting_indices=indices[:96],heldout_indices=indices[96:],split='128 evenly spaced TRAIN frames; first96 fit, last32 held out; no validation',
            features='128 source encoder channels + 32 source IF drive channels at sent sites; no ego/GT/receiver state',
            fitting='train-only feature normalization; per-channel sign-balanced ridge targets sign(detection_loss/count); lambda=0.001; bias unpenalized',
            selection='up to32 predicted-positive unit events in baseline prefix; suppress latest copies, then ordinary true-byte prefix admission with refill',
            random='same number of suppressions as student, uniform baseline unit events, seed+frame index',
            oracle='held-out labels for diagnostic only, up to32 positive-gradient suppressions, never affects fit/student choice',
            budget_bytes=2048,augmentation=[],all_detector_state_frozen=True,
            gate='student mean actual loss below baseline and matched random; beats random on >=20/32 frames; every packet <=2048bytes',
            exit='one fixed 96/32 linear probe, no hidden width/lambda/seed/deletion-size sweep; no validation AP or automatic detector training',
            git_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip())
        write_json(root/'protocol.json',protocol)
        xs=[];ys=[];ms=[];records=[];training=[];probe=None;started=time.time()
        for position,index in enumerate(indices):
            if time.time()-started>7200:raise TimeoutError('Two-hour probe cap')
            if position==96:
                write_json(root/'status.json',dict(state='fitting_source_only',frames=96,total=128))
                probe=fit_probe(np.concatenate(xs),np.concatenate(ys),np.concatenate(ms))
                np.savez_compressed(str(root/'source_probe.npz'),**probe)
                del xs,ys,ms
            batch=train_utils.to_device(data.collate_batch_train([data[index]]),torch.device('cuda'))
            output,loss=checked_forward(model,criterion,batch);c=captured['counts']
            gradient=torch.autograd.grad(loss,c)[0];assert torch.isfinite(gradient).all()
            counts=c.detach().cpu().numpy();g=gradient.detach().cpu().numpy()
            rows=np.flatnonzero(np.any(counts.reshape(32,-1)!=0,axis=0))
            features=torch.cat((captured['source_features'],captured['drive']),dim=1)[0].reshape(160,-1).T[rows].cpu().numpy()
            target=g.reshape(32,-1).T[rows];mask=counts.reshape(32,-1).T[rows]>0
            folder=root/('fit' if position<96 else 'heldout')/str(index);folder.mkdir(parents=True)
            np.savez_compressed(str(folder/'features_teacher.npz'),features=features,gradient=target,observed=mask,spatial_rows=rows)
            if position<96:
                xs.append(features);ys.append(target);ms.append(mask)
                training.append(dict(index=index,frame_id=str(data.data[index]),rows=len(rows),file=str((folder/'features_teacher.npz').relative_to(root)),sha256=sha(folder/'features_teacher.npz')))
            else:
                # Scores receive only sender-local tensors and the frozen fitted probe.
                score=np.zeros_like(counts,dtype=np.float64);score.reshape(32,-1)[:,rows]=source_scores(features,probe).T
                deletions,k=delete_events(counts,score,seed+index)
                oracle,_=delete_events(counts,g,seed+index)
                removals=dict(student=deletions['oracle']['removed'],random=deletions['random']['removed'],oracle=oracle['oracle']['removed'])
                stream=integrate_and_fire(captured['drive'],8);full=stream.prefix(len(stream.addresses)).detach().cpu().numpy()
                contract=captured['contract'];meta=dict(contract=asdict(contract),source_to_ego=captured['pose'][0].cpu().tolist(),family='if8',schema='task-count32-v1')
                base,selected,admission=wire.select(full,np.ones(32,dtype=np.float32),meta,2048,addresses=stream.addresses)
                assert np.array_equal(selected,counts) and len(base)==output['packet_bytes'][0] and hashlib.sha256(base).hexdigest()==model.snapshot_records[0]['packet_sha256']
                (folder/'baseline.bin').write_bytes(base)
                row=dict(index=index,frame_id=str(data.data[index]),baseline_loss=float(loss.detach()),baseline_bytes=len(base),student_suppressed_events=k,variants={})
                with torch.no_grad():
                    for name,removed in removals.items():
                        packet,selected,record=select_suppressed(full,stream.addresses,admission['accepted_units'],removed,meta,2048)
                        decoded,paid=wire.decode(packet,meta);assert np.array_equal(decoded,selected) and np.all(paid==1)
                        q=native(captured['membrane'],c.new_tensor(decoded),captured['pose'],contract,captured['receiver'],captured['decode'])
                        prediction=model.detect_bev(q.flatten(1,2));current,_=detection_loss(prediction,batch['ego']['label_dict'],criterion,0.)
                        assert torch.isfinite(current);(folder/(name+'.bin')).write_bytes(packet)
                        row['variants'][name]=dict(loss=float(current),loss_delta=float(current)-row['baseline_loss'],bytes=len(packet),
                            selected_events=record['selected_events'],suppressed_events=int(removed.sum()),sha256=hashlib.sha256(packet).hexdigest())
                np.savez_compressed(str(folder/'selection.npz'),original_counts=counts,scores=score,student_removed=removals['student'],random_removed=removals['random'],oracle_removed=removals['oracle'])
                records.append(row)
            write_json(root/'status.json',dict(state='collecting_teacher' if position<96 else 'heldout_evaluation',frames=position+1,total=128))
            del output,loss,gradient,c
        assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items()) and sha(checkpoint)==DIGEST
        means={name:float(np.mean([r['baseline_loss'] if name=='baseline' else r['variants'][name]['loss'] for r in records])) for name in ('baseline','student','random','oracle')}
        beats=sum(r['variants']['student']['loss']<r['variants']['random']['loss'] for r in records)
        bytes_ok=all(r['baseline_bytes']<=2048 and all(v['bytes']<=2048 for v in r['variants'].values()) for r in records)
        gate=means['student']<min(means['baseline'],means['random']) and beats>=20 and bytes_ok
        write_json(root/'fitting_index.json',dict(frames=training,source_probe_sha256=sha(root/'source_probe.npz')))
        write_json(root/'summary.json',dict(passed=True,all_detector_state_unchanged=True,fit_frames=96,heldout_frames=32,means=means,
            student_beats_random_frames=beats,hypothesis_gate=gate,all_packets_within_budget=bytes_ok,
            source_probe_sha256=sha(root/'source_probe.npz'),records=records,protocol=protocol))
        write_json(root/'status.json',dict(state='complete',summary_sha256=sha(root/'summary.json')))
    except Exception:
        write_json(root/'status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry'));raise
    finally:
        adapter.stream_readout=native
        if hook is not None:hook.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--launch',action='store_true');p.add_argument('--gpu',type=int,default=0);a=p.parse_args();root=Path(a.root)
    if a.launch:
        from opencood.tools.launch_e3dsnn_joint_round import check_gpu
        if root.exists():raise FileExistsError(str(root))
        admission=check_gpu(a.gpu,True,.2)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with Path(str(root)+'.log').open('xb') as f:
            proc=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.probe_e3dsnn_source_utility','--root',str(root),'--source',a.source],cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(Path(str(root)+'_launch.json'),dict(pid=proc.pid,gpu=a.gpu,admission=admission,started_unix=time.time()))
        print(json.dumps(dict(pid=proc.pid,root=str(root))))
    else:run(root,Path(a.source))
