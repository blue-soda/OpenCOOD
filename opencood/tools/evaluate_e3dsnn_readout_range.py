"""Paired frozen Count4 versus uncapped counts, sharing source/packet/current."""
import argparse
import copy
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
from opencood.tools.launch_e3dsnn_hybrid_round import REPO,sha,read,evaluation
from opencood.tools.train_e3dsnn_single import write_json,seed_worker
from opencood.tools.profile_e3dsnn_snapshot_charge import SOURCE,DIGEST
from opencood.models.fuse_modules.e3dsnn_readout_range import paired_count_readouts

MODES=('b2048_capped','b2048_uncapped','full_capped','full_uncapped')


def run(root,limit):
    from torch.utils.data import DataLoader
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.utils import eval_utils,eval_3d_utils,task_snapshot_packet as wire
    from opencood.utils.e3dsnn_transplant import encoder_digest
    from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire,stream_readout
    from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract,count_readout
    from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume
    root.mkdir(parents=True,exist_ok=False)
    try:
        checkpoint=SOURCE/'train/best_ap70.pth';config=SOURCE/'train.yaml'
        assert sha(checkpoint)==DIGEST
        random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
        torch.cuda.set_per_process_memory_fraction(.2)
        cfg=load_yaml(str(config));data=build_dataset(cfg,train=False);assert len(data)==1738
        loader=DataLoader(data,batch_size=1,shuffle=False,num_workers=2,collate_fn=data.collate_batch_test,
            worker_init_fn=seed_worker,multiprocessing_context='spawn')
        model=train_utils.create_model(cfg).cuda().eval();initial=torch.load(str(checkpoint),map_location='cpu')
        model.load_state_dict(initial,strict=True);model.requires_grad_(False);model.packet_roundtrip=True
        encoder=encoder_digest(model);basis=model.event_basis_digest()
        stats={m:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.3,.5,.7)} for m in MODES}
        stats3d=copy.deepcopy(stats);frames={m:[] for m in MODES};audit=[]
        protocol=dict(checkpoint=str(checkpoint),checkpoint_sha256=DIGEST,config_sha256=sha(config),
            data_manifest_sha256=sha(Path(cfg['data_manifest'])),split_sha256={k:sha(Path(cfg[k])) for k in ('root_dir','validate_dir')},
            seed=20260929,limit=limit,training=False,modes=MODES,cuda_allocator_memory_fraction=.2,
            shared_source_packet_and_current=True,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
            intervention='Remove only receiver output ceiling on original remote-supported spatial sites',
            unchanged='sender max8, IF ordering, bytes, geometry, local-only path, D/gain, BEV/head and all weights',
            exit='one paired 2048/full comparison, no ceiling/gain/temperature sweep or automatic retraining')
        write_json(root/'protocol.json',protocol)
        for mode in MODES:
            folder=root/mode;(folder/'predictions').mkdir(parents=True);(folder/'packets').mkdir()
            write_json(folder/'manifest_range.json',dict(protocol,mode=mode))
        started=time.time()
        with torch.no_grad():
            for i,batch in enumerate(loader):
                if limit and i>=limit:break
                if time.time()-started>14400:raise TimeoutError('Paired evaluation four-hour cap')
                batch=train_utils.to_device(batch,torch.device('cuda'));item=batch['ego']
                local,membrane=model.encode_membrane(item['processed_lidar'],1)
                road=model.encode_lidar(item['road_lidar'],1)
                stream=integrate_and_fire(torch.relu(model.content.project(road.dense())),8)
                full=stream.prefix(len(stream.addresses));scales=np.ones(32,dtype=np.float32)
                contract=MessageContract(str(item['frame_ids'][0]),str(item['road_frame_ids'][0]),
                    'infrastructure',encoder,'task32-'+basis,tuple(int(x) for x in full.shape[2:]),
                    tuple(model.origin),tuple(model.step),tuple(int(x) for x in item['timestamps_us'][0].cpu().tolist()))
                pose=item['road_to_ego'];meta=dict(contract=asdict(contract),source_to_ego=pose[0].cpu().tolist(),
                    family='if8',schema='task-count32-v1')
                if limit:
                    zero=torch.zeros_like(full)
                    a,b=paired_count_readouts(membrane,torch.zeros_like(membrane),torch.zeros_like(membrane[:,:1],dtype=torch.bool))
                    native=stream_readout(membrane,zero,pose,contract,model.joint_receiver,model.content.decode)
                    assert torch.equal(a,b) and torch.equal(a,native) and torch.equal(a,local.dense())
                source_sha=hashlib.sha256(stream.addresses.tobytes()).hexdigest()
                for budget,label in ((2048,'b2048'),(None,'full')):
                    packet,selected,record=wire.select(full.cpu().numpy(),scales,meta,budget,addresses=stream.addresses)
                    decoded,paid=wire.decode(packet,meta);assert np.array_equal(selected,decoded) and np.all(paid==1)
                    counts=full.new_tensor(decoded)
                    expected=full if budget is None else stream.prefix(record['accepted_units'])
                    assert torch.equal(counts,expected)
                    aligned,coverage=warp_volume(counts,pose,contract.origin_xyz,contract.step_xyz,geometry_dtype=torch.float64)
                    gain=2*torch.sigmoid(model.joint_receiver.local_gate(membrane))*model.joint_receiver.current_scale
                    current=model.content.decode(aligned)*gain
                    supported=(aligned!=0).any(1,keepdim=True)&(coverage>0)
                    capped,uncapped=paired_count_readouts(membrane,current,supported)
                    native=stream_readout(membrane,counts,pose,contract,model.joint_receiver,model.content.decode)
                    assert torch.equal(native,capped)
                    assert torch.equal(uncapped,torch.where(supported,torch.floor((membrane+current).clamp(min=0)+.5),count_readout(membrane)))
                    assert torch.all(uncapped>=capped)
                    outputs={kind:model.detect_bev(state.flatten(1,2)) for kind,state in (('capped',capped),('uncapped',uncapped))}
                    original_exact=None
                    if limit:
                        old_local,old_road=model.encode_membrane,model.encode_lidar
                        try:
                            model.encode_membrane=lambda lidar,n:(local,membrane);model.encode_lidar=lambda lidar,n:road
                            model.snapshot_budget=budget;original=model(item)
                            original_exact=all(torch.equal(original[k],outputs['capped'][k]) for k in ('psm','rm','dm'))
                            assert original_exact and original['packet_bytes']==[len(packet)]
                        finally:model.encode_membrane,model.encode_lidar=old_local,old_road
                    for kind,output in outputs.items():
                        mode=label+'_'+kind;folder=root/mode
                        boxes,scores,gt=data.post_process_no_fusion(batch,{'ego':output})
                        for threshold in stats[mode]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[mode],threshold)
                        eval_3d_utils.calculate_tp_fp_3d(boxes,scores,gt,stats3d[mode])
                        pred=boxes.cpu().numpy() if boxes is not None else np.empty((0,8,3),dtype=np.float32)
                        score=scores.cpu().numpy() if scores is not None else np.empty(0,dtype=np.float32)
                        assert np.isfinite(pred).all() and np.isfinite(score).all()
                        fid=contract.receiver_frame;assert fid==str(data.data[i])
                        path=folder/'predictions'/('{:06d}.npz'.format(i))
                        np.savez_compressed(str(path),frame_id=np.asarray(fid),pred_boxes=pred,scores=score,gt_boxes=gt.cpu().numpy())
                        (folder/'packets'/('{:06d}.bin'.format(i))).write_bytes(packet)
                        frames[mode].append(dict(index=i,frame_id=fid,file=path.name,sha256=sha(path),predicted_boxes=len(pred),gt_boxes=len(gt),
                            communication=dict(road_to_vehicle_packet_bytes=[len(packet)],total_bytes=len(packet))))
                    audit.append(dict(index=i,frame_id=contract.receiver_frame,budget=budget,bytes=len(packet),
                        packet_sha256=hashlib.sha256(packet).hexdigest(),source_events_sha256=source_sha,
                        original_readout_exact=True,original_heads_exact=original_exact,intervention_formula_exact=True,
                        changed_readout_values=int((capped!=uncapped).sum()),max_uncapped=float(uncapped.max())))
                if (i+1)%25==0:write_json(root/'status.json',dict(state='running',frames=i+1,total=len(data)))
        assert len(audit)==2*(limit or 1738)
        assert sha(checkpoint)==DIGEST and all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
        write_json(root/'paired_audit.json',dict(passed=True,all_state_unchanged=True,records=audit))
        results={}
        for mode in MODES:
            folder=root/mode;rows=frames[mode];lengths=[f['communication']['total_bytes'] for f in rows]
            if mode.startswith('b2048'):assert max(lengths)<=2048
            write_json(folder/'predictions/index.json',dict(schema=1,frames=rows,order='validation loader order'))
            metrics=dict(samples=len(rows),indexed_samples=len(data),full_split=len(rows)==len(data),
                predicted_boxes=sum(f['predicted_boxes'] for f in rows),gt_boxes=sum(f['gt_boxes'] for f in rows),
                ap={str(t):eval_utils.calculate_ap(stats[mode],t)[0] for t in stats[mode]},
                ap_3d={str(t):eval_utils.calculate_ap(stats3d[mode],t)[0] for t in stats3d[mode]},
                metric='BEV and upright 3D AP; shared source/packet/current; output ceiling only',
                predictions=dict(directory='predictions',frames=len(rows),index_sha256=sha(folder/'predictions/index.json')),
                feature_packets=dict(count=len(rows),mean_bytes=float(np.mean(lengths)),median_bytes=float(np.median(lengths)),
                    min_bytes=min(lengths),max_bytes=max(lengths),p95_bytes=float(np.percentile(lengths,95)),total_bytes=sum(lengths),
                    scope='serialized payload and full metadata; excludes network framing/retransmission'))
            write_json(folder/'evaluation.json',metrics)
            write_json(root/'status.json',dict(state='cpu_replay',mode=mode))
            with (root/(mode+'_replay.log')).open('xb') as log:
                subprocess.run([sys.executable,'-u','-m','opencood.tools.replay_e3dsnn_predictions','--evaluation-dir',str(folder)],
                    cwd=str(REPO),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
            if limit:
                replay=read(folder/'offline_replay.json');assert replay['passed'] and replay['exact_match'] and replay['all_file_hashes_verified']
                results[mode]=dict(evaluation=metrics,replay=replay)
            else:results[mode]=evaluation(folder)
        for label in ('b2048','full'):
            a,b=[root/(label+'_'+k) for k in ('capped','uncapped')]
            assert results[label+'_capped']['evaluation']['feature_packets']==results[label+'_uncapped']['evaluation']['feature_packets']
            for i in range(limit or 1738):
                name='{:06d}.bin'.format(i);assert (a/'packets'/name).read_bytes()==(b/'packets'/name).read_bytes()
        write_json(root/'summary.json',dict(passed=True,protocol=protocol,results=results,
            all_state_unchanged=True,paired_audit_sha256=sha(root/'paired_audit.json')))
        write_json(root/'status.json',dict(state='complete',summary_sha256=sha(root/'summary.json')))
    except Exception:
        write_json(root/'status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry'));raise


def coordinate(root,gpu):
    from opencood.tools.launch_e3dsnn_joint_round import check_gpu
    try:
        for stage,limit in (('smoke',2),('full',0)):
            write_json(root/(stage+'_admission.json'),check_gpu(gpu,True,.2))
            write_json(root/'pipeline_status.json',dict(state='running',stage=stage))
            with (root/(stage+'.log')).open('xb') as log:
                subprocess.run([sys.executable,'-u','-m','opencood.tools.evaluate_e3dsnn_readout_range',
                    '--root',str(root/stage),'--limit',str(limit)],cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    check=True,timeout=21600)
            assert read(root/stage/'summary.json')['passed']
        write_json(root/'pipeline_status.json',dict(state='complete',summary_sha256=sha(root/'full/summary.json')))
    except Exception:
        write_json(root/'pipeline_status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry'));raise


def launch(root,gpu):
    from opencood.tools.launch_e3dsnn_joint_round import check_gpu
    if root.exists():raise FileExistsError('Fresh root required')
    assert sha(SOURCE/'train/best_ap70.pth')==DIGEST
    admission=check_gpu(gpu,True,.2);root.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/'pipeline.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.evaluate_e3dsnn_readout_range',
            '--root',str(root),'--gpu',str(gpu),'--coordinate'],cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,gpu=gpu,admission=admission,started_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),training=False)
    write_json(root/'launch.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--launch',action='store_true');p.add_argument('--coordinate',action='store_true');p.add_argument('--limit',type=int,choices=(0,2),default=0)
    a=p.parse_args()
    if a.launch:launch(Path(a.root),a.gpu)
    elif a.coordinate:coordinate(Path(a.root),a.gpu)
    else:run(Path(a.root),a.limit)
