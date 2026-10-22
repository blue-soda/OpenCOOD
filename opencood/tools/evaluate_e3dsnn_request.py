"""Frozen paired request/no-request AP evaluation with paid bidirectional bytes."""
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
from opencood.tools.launch_e3dsnn_hybrid_round import sha,read,evaluation
from opencood.tools.train_e3dsnn_single import write_json,seed_worker
from opencood.tools.profile_e3dsnn_snapshot_charge import DIGEST
from opencood.models.fuse_modules.e3dsnn_request import request_codes,encode_request,decode_request,request_on_sender,request_order,POLICY

REPO=Path(__file__).resolve().parents[2]
MODES=tuple('b{}_{}'.format(b,k) for b in (2048,4096) for k in ('native','blind','query'))


def run(root,source,limit):
    from torch.utils.data import DataLoader
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.utils import eval_utils,eval_3d_utils,task_snapshot_packet as wire
    from opencood.utils.e3dsnn_transplant import encoder_digest
    from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire,stream_readout
    from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
    root.mkdir(parents=True,exist_ok=False)
    try:
        checkpoint=source/'train/best_ap70.pth';assert sha(checkpoint)==DIGEST
        random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
        torch.cuda.set_per_process_memory_fraction(.2)
        cfg=load_yaml(str(source/'train.yaml'));data=build_dataset(cfg,train=False);assert len(data)==1738
        loader=DataLoader(data,batch_size=1,shuffle=False,num_workers=2,collate_fn=data.collate_batch_test,worker_init_fn=seed_worker,multiprocessing_context='spawn')
        model=train_utils.create_model(cfg).cuda().eval();initial=torch.load(str(checkpoint),map_location='cpu')
        model.load_state_dict(initial,strict=True);model.requires_grad_(False);model.packet_roundtrip=True
        encoder=encoder_digest(model);basis=model.event_basis_digest()
        stats={m:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.3,.5,.7)} for m in MODES};stats3d=copy.deepcopy(stats)
        frames={m:[] for m in MODES};audits=[]
        protocol=dict(checkpoint_sha256=DIGEST,source=str(source),config_sha256=sha(source/'train.yaml'),
            data_manifest_sha256=sha(Path(cfg['data_manifest'])),split_sha256={k:sha(Path(cfg[k])) for k in ('root_dir','validate_dir')},
            seed=20260929,limit=limit,training=False,modes=MODES,request_policy=POLICY,
            paired='one local/road encoding and full event multiset per frame, all modes',
            budgets='2048 and4096, reverse request plus forward response including both full metadata envelopes',
            blind='transmit same actual request then ignore it; native IF reply with reduced remaining budget',
            semantics='scalar spatial request, no semantic channel addition; ego-to-road inverse warp; no GT request inputs',
            gate='query AP70 >= native and blind +0.1pp at at least one registered budget; report both; only promotes further study, not overall success',
            exit='one fixed heuristic and two budgets; no query threshold/block/bit depth/weight sweep or automatic training',
            code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip())
        write_json(root/'protocol.json',protocol)
        for mode in MODES:
            folder=root/mode;(folder/'predictions').mkdir(parents=True);(folder/'packets').mkdir();(folder/'requests').mkdir()
            write_json(folder/'manifest_request.json',dict(protocol,mode=mode))
        started=time.time()
        with torch.no_grad():
            for i,batch in enumerate(loader):
                if limit and i>=limit:break
                if time.time()-started>18000:raise TimeoutError('Five-hour inference limit')
                batch=train_utils.to_device(batch,torch.device('cuda'));item=batch['ego']
                local,membrane=model.encode_membrane(item['processed_lidar'],1);road=model.encode_lidar(item['road_lidar'],1)
                stream=integrate_and_fire(torch.relu(model.content.project(road.dense())),8);full=stream.prefix(len(stream.addresses));scales=np.ones(32,dtype=np.float32)
                contract=MessageContract(str(item['frame_ids'][0]),str(item['road_frame_ids'][0]),'infrastructure',encoder,'task32-'+basis,
                    tuple(int(x) for x in full.shape[2:]),tuple(model.origin),tuple(model.step),tuple(int(x) for x in item['timestamps_us'][0].cpu().tolist()))
                pose=item['road_to_ego'];meta=dict(contract=asdict(contract),source_to_ego=pose[0].cpu().tolist(),family='if8',schema='task-count32-v1')
                codes=request_codes(membrane);request=encode_request(codes,meta);decoded_request=decode_request(request,meta)
                assert np.array_equal(codes,decoded_request)
                priorities,coverage=request_on_sender(decoded_request,meta,full.device)
                query_ids=request_order(stream.addresses,stream.times,priorities.cpu().numpy())
                assert np.array_equal(np.sort(query_ids),np.sort(stream.addresses))
                if not request:assert np.array_equal(query_ids,stream.addresses)
                native_exact=[]
                for budget in (2048,4096):
                    assert len(request)<=budget,'Request alone exceeds total budget'
                    for kind in ('native','blind','query'):
                        reverse=b'' if kind=='native' else request
                        ids=query_ids if kind=='query' else stream.addresses
                        packet,selected,record=wire.select(full.cpu().numpy(),scales,meta,budget-len(reverse),addresses=ids)
                        restored,paid=wire.decode(packet,meta);assert np.array_equal(selected,restored) and np.all(paid==1)
                        counts=full.new_tensor(restored)
                        state=stream_readout(membrane,counts,pose,contract,model.joint_receiver,model.content.decode)
                        output=model.detect_bev(state.flatten(1,2))
                        if limit and kind=='native':
                            old_local,old_road=model.encode_membrane,model.encode_lidar
                            try:
                                model.encode_membrane=lambda lidar,n:(local,membrane);model.encode_lidar=lambda lidar,n:road
                                model.snapshot_budget=budget;original=model(item)
                                assert all(torch.equal(original[k],output[k]) for k in ('psm','rm','dm')) and original['packet_bytes']==[len(packet)]
                                native_exact.append(budget)
                            finally:model.encode_membrane,model.encode_lidar=old_local,old_road
                        mode='b{}_{}'.format(budget,kind);folder=root/mode
                        boxes,scores,gt=data.post_process_no_fusion(batch,{'ego':output})
                        for t in stats[mode]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[mode],t)
                        eval_3d_utils.calculate_tp_fp_3d(boxes,scores,gt,stats3d[mode])
                        pred=boxes.cpu().numpy() if boxes is not None else np.empty((0,8,3),dtype=np.float32)
                        score=scores.cpu().numpy() if scores is not None else np.empty(0,dtype=np.float32)
                        assert np.isfinite(pred).all() and np.isfinite(score).all() and contract.receiver_frame==str(data.data[i])
                        path=folder/'predictions'/('{:06d}.npz'.format(i));np.savez_compressed(str(path),frame_id=np.asarray(contract.receiver_frame),pred_boxes=pred,scores=score,gt_boxes=gt.cpu().numpy())
                        name='{:06d}.bin'.format(i);(folder/'packets'/name).write_bytes(packet);(folder/'requests'/name).write_bytes(reverse)
                        total=len(packet)+len(reverse);assert total<=budget
                        frames[mode].append(dict(index=i,frame_id=contract.receiver_frame,file=path.name,sha256=sha(path),predicted_boxes=len(pred),gt_boxes=len(gt),
                            communication=dict(road_to_vehicle_packet_bytes=[len(packet)],vehicle_to_road_request_bytes=[len(reverse)],total_bytes=total),
                            forward_sha256=hashlib.sha256(packet).hexdigest(),reverse_sha256=hashlib.sha256(reverse).hexdigest()))
                audits.append(dict(index=i,frame_id=contract.receiver_frame,request_bytes=len(request),request_sha256=hashlib.sha256(request).hexdigest(),
                    request_roundtrip_exact=True,event_multiset_unchanged=True,smoke_native_exact_budgets=native_exact,
                    requested_source_sites=int((priorities>0).sum()),covered_source_sites=int((coverage>0).sum())))
                if (i+1)%25==0:write_json(root/'status.json',dict(state='running',frames=i+1,total=len(data)))
        assert len(audits)==(limit or 1738)
        assert sha(checkpoint)==DIGEST and all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
        write_json(root/'paired_audit.json',dict(passed=True,all_state_unchanged=True,records=audits))
        results={}
        for mode in MODES:
            folder=root/mode;rows=frames[mode];lengths=[f['communication']['total_bytes'] for f in rows]
            write_json(folder/'predictions/index.json',dict(schema=1,frames=rows,order='validation loader order'))
            metrics=dict(samples=len(rows),indexed_samples=len(data),full_split=len(rows)==len(data),predicted_boxes=sum(f['predicted_boxes'] for f in rows),gt_boxes=sum(f['gt_boxes'] for f in rows),
                ap={str(t):eval_utils.calculate_ap(stats[mode],t)[0] for t in stats[mode]},ap_3d={str(t):eval_utils.calculate_ap(stats3d[mode],t)[0] for t in stats3d[mode]},
                metric='BEV and upright3D AP; both communication directions paid',predictions=dict(directory='predictions',frames=len(rows),index_sha256=sha(folder/'predictions/index.json')),
                feature_packets=dict(count=len(rows),mean_bytes=float(np.mean(lengths)),median_bytes=float(np.median(lengths)),min_bytes=min(lengths),max_bytes=max(lengths),p95_bytes=float(np.percentile(lengths,95)),total_bytes=sum(lengths),
                    mean_reverse_bytes=float(np.mean([f['communication']['vehicle_to_road_request_bytes'][0] for f in rows])),scope='forward+reverse serialized payload and complete metadata; excludes network framing/retransmission'))
            write_json(folder/'evaluation.json',metrics);write_json(root/'status.json',dict(state='cpu_replay',mode=mode))
            with (root/(mode+'_replay.log')).open('xb') as log:
                subprocess.run([sys.executable,'-u','-m','opencood.tools.replay_e3dsnn_predictions','--evaluation-dir',str(folder)],cwd=str(REPO),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
            if limit:
                replay=read(folder/'offline_replay.json');assert replay['passed'] and replay['exact_match'] and replay['all_file_hashes_verified'];results[mode]=dict(evaluation=metrics,replay=replay)
            else:results[mode]=evaluation(folder)
        write_json(root/'summary.json',dict(passed=True,all_state_unchanged=True,protocol=protocol,results=results,paired_audit_sha256=sha(root/'paired_audit.json')))
        write_json(root/'status.json',dict(state='complete',summary_sha256=sha(root/'summary.json')))
    except Exception:
        write_json(root/'status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry'));raise


def coordinate(root,source,gpu):
    from opencood.tools.launch_e3dsnn_joint_round import check_gpu
    try:
        for stage,limit in (('smoke',2),('full',0)):
            write_json(root/(stage+'_admission.json'),check_gpu(gpu,True,.2));write_json(root/'pipeline_status.json',dict(state='running',stage=stage))
            with (root/(stage+'.log')).open('xb') as log:
                subprocess.run([sys.executable,'-u','-m','opencood.tools.evaluate_e3dsnn_request','--root',str(root/stage),'--source',str(source),'--limit',str(limit)],cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=28800)
            assert read(root/stage/'summary.json')['passed']
        write_json(root/'pipeline_status.json',dict(state='complete',summary_sha256=sha(root/'full/summary.json')))
    except Exception:
        write_json(root/'pipeline_status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry'));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--launch',action='store_true');p.add_argument('--coordinate',action='store_true');p.add_argument('--limit',type=int,choices=(0,2),default=0);a=p.parse_args();root=Path(a.root);source=Path(a.source)
    if a.launch:
        from opencood.tools.launch_e3dsnn_joint_round import check_gpu
        if root.exists():raise FileExistsError(str(root))
        assert sha(source/'train/best_ap70.pth')==DIGEST
        admission=check_gpu(a.gpu,True,.2);root.mkdir(parents=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (root/'pipeline.log').open('xb') as f:
            proc=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.evaluate_e3dsnn_request','--root',str(root),'--source',str(source),'--gpu',str(a.gpu),'--coordinate'],cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(root/'launch.json',dict(pid=proc.pid,gpu=a.gpu,admission=admission,started_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()));print(json.dumps(dict(pid=proc.pid,root=str(root))))
    elif a.coordinate:coordinate(root,source,a.gpu)
    else:run(root,source,a.limit)
