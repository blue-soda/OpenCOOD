"""Frozen strong ANN engineering reference under a common scaled snapshot codec.

Historical training, receiver and selection differ: NOT an activation-only
causal ablation. Full-message source equivalence precedes any full-split AP.
"""
import argparse
from dataclasses import asdict, replace
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
from opencood.utils import task_snapshot_packet as wire
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, evaluation
from opencood.tools.train_e3dsnn_single import validate, write_json

BASE=Path('/data1/tzm/workspace/diagnostics')
SOURCES={
    'if8': (BASE/'e3dsnn_event_receiver_20261021/receiver/train/best_ap70.pth',
        BASE/'e3dsnn_event_receiver_20261021/receiver/b4096.yaml',
        '60b7e0ff84e992862902aa9325b56ac13ab5658d70c4aacb194028638f5105dc'),
    'ann2bit': (BASE/'e3dsnn_content_round1_v2_20261019/ann_learn_adapt10/best_ap70.pth',
        BASE/'e3dsnn_content_round1_v2_20261019/ann_learn/full.yaml',
        'd0dc7b9ab4ccaad531a8fc63e0a7e8b91a62ab5fa7cf4d0740e366c6422b4bd8')}
MODES=('full','b2048')


def ann_readout(model,ego,counts,scales,contract,pose):
    from opencood.models.fuse_modules.e3dsnn_content_receiver import ContentAccumulator
    from opencood.models.fuse_modules.e3dsnn_readout import residual_readout
    basis=model.content.basis_digest()
    native=replace(contract,channel_basis='content32-'+basis)
    receiver=ContentAccumulator(basis,'ann2bit')
    state=receiver.begin(counts,native,pose,scales)
    for k in (1,2,3):
        event=(counts>=k).to(ego.dtype)
        latent,coverage=receiver.receive(state,event,native,pose,scales,k)
    nonempty=(state.source_counts.detach().abs().sum()>0).to(ego.dtype)
    mask=(coverage*nonempty).flatten(1,2);supported=mask.amax(1,keepdim=True)
    local=ego.flatten(1,2);remote=model.content.decode(latent).flatten(1,2)
    delta=model.fusion_net(torch.cat((local,remote,mask),1))
    residual=model.fusion_scale*supported*delta
    fused=residual_readout(local,residual,supported,model.fusion_activation,model.fusion_readout)
    fused=torch.where(supported>0,fused,local)
    receiver.finish(state)
    return latent,fused


class FrozenSnapshot(torch.nn.Module):
    packet_roundtrip=True

    def __init__(self,model,family,mode,folder,original_check):
        super().__init__()
        from opencood.utils.e3dsnn_transplant import encoder_digest
        self.model=model;self.family=family;self.mode=mode;self.folder=folder
        self.original_check=original_check;self.records=[];self.encoder=encoder_digest(model)
        self.basis=model.event_basis_digest() if family=='if8' else model.content.basis_digest()

    def forward(self,data):
        from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
        from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire,stream_readout
        if self.training or data['object_bbx_mask'].shape[0]!=1:raise ValueError('Eval-only single frame')
        model=self.model;tic=time.perf_counter()
        local_sparse,membrane=model.encode_membrane(data['processed_lidar'],1)
        ego=local_sparse.dense();road_sparse=model.encode_lidar(data['road_lidar'],1)
        if self.family=='if8':
            stream=integrate_and_fire(torch.relu(model.content.project(road_sparse.dense())),8)
            full=stream.prefix(len(stream.addresses));scales=np.ones(32,dtype=np.float32)
            order=dict(addresses=stream.addresses);message=None
        else:
            message=model.content.encode(model.select_road(road_sparse,1));full=message.counts
            scales=message.scales.detach().cpu().numpy().reshape(32)
            positions=road_sparse.indices[:,1:].cpu().numpy();shape=full.shape[2:]
            flat=(positions[:,0]*shape[1]+positions[:,1])*shape[2]+positions[:,2]
            score=road_sparse.features.detach().cpu().numpy().sum(1,dtype=np.float64)
            rank=np.lexsort((flat,-score));active=np.flatnonzero(np.any(full.cpu().numpy().reshape(32,-1)!=0,axis=0))
            order=dict(row_order=flat[rank][np.isin(flat[rank],active)])
        contract=MessageContract(str(data['frame_ids'][0]),str(data['road_frame_ids'][0]),
            'infrastructure',self.encoder,'task32-'+self.basis,
            tuple(int(x) for x in full.shape[2:]),tuple(model.origin),tuple(model.step),
            tuple(int(x) for x in data['timestamps_us'][0].cpu().tolist()))
        pose=data['road_to_ego'];meta=dict(contract=asdict(contract),source_to_ego=pose[0].cpu().tolist(),
            family=self.family,schema='task-count32-v1')
        full_array=full.cpu().numpy();budget=None if self.mode=='full' else 2048
        packet,selected,record=wire.select(full_array,scales,meta,budget,**order)
        decoded,wire_scales=wire.decode(packet,meta)
        assert np.array_equal(selected,decoded)
        direct=full.new_tensor(selected);counts=full.new_tensor(decoded)
        if self.family=='if8':
            if self.mode=='b2048':assert torch.equal(direct,stream.prefix(record['accepted_units']))
            a=stream_readout(membrane,direct,pose,contract,model.joint_receiver,model.content.decode)
            b=stream_readout(membrane,counts,pose,contract,model.joint_receiver,model.content.decode)
            assert torch.equal(a,b);direct_output=model.detect_bev(a.flatten(1,2));output=model.detect_bev(b.flatten(1,2))
        else:
            assert torch.equal(direct,full*(direct!=0))
            a,af=ann_readout(model,ego,direct,full.new_tensor(scales).reshape(1,32,1,1,1),contract,pose)
            b,bf=ann_readout(model,ego,counts,full.new_tensor(wire_scales).reshape(1,32,1,1,1),contract,pose)
            assert torch.equal(a,b) and torch.equal(af,bf)
            direct_output=model.detect_bev(af);output=model.detect_bev(bf)
        assert all(torch.equal(output[k],direct_output[k]) for k in ('psm','rm','dm'))
        original_exact=None
        if self.original_check and self.mode=='full':
            # Compare the real source forward using this exact encoded frame.
            old_local,old_road=model.encode_membrane,model.encode_lidar
            old_encode=model.content.encode;old_wire=model.packet_roundtrip
            old_budget=getattr(model,'event_budget',None)
            try:
                model.encode_membrane=lambda lidar,n:(local_sparse,membrane)
                model.encode_lidar=lambda lidar,n:road_sparse
                if message is not None:model.content.encode=lambda features:message
                model.packet_roundtrip=True
                if self.family=='if8':model.event_budget=2**30
                native=model(data)
                original_exact=all(torch.equal(output[k],native[k]) for k in ('psm','rm','dm'))
                assert original_exact,'Original full source differs from snapshot reconstruction'
            finally:
                model.encode_membrane,model.encode_lidar=old_local,old_road
                model.content.encode=old_encode;model.packet_roundtrip=old_wire
                if old_budget is not None:model.event_budget=old_budget
        name='{:06d}.bin'.format(len(self.records));(self.folder/'packets'/name).write_bytes(packet)
        record.update(index=len(self.records),frame_id=contract.receiver_frame,
            packet_file=name,packet_sha256=hashlib.sha256(packet).hexdigest(),
            full_counts_sha256=hashlib.sha256(full_array.tobytes()).hexdigest(),
            full_scales_sha256=hashlib.sha256(scales.tobytes()).hexdigest(),
            direct_wire_exact=True,original_full_exact=original_exact,
            audit_forward_seconds=time.perf_counter()-tic)
        self.records.append(record)
        return dict(output,packet_bytes=[len(packet)])


def execute(folder,family,mode,limit):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    checkpoint,config,digest=SOURCES[family]
    assert sha(checkpoint)==digest
    random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    torch.cuda.set_per_process_memory_fraction(.2)
    cfg=load_yaml(str(config));dataset=build_dataset(cfg,train=False)
    model=train_utils.create_model(cfg).cuda().eval();initial=torch.load(str(checkpoint),map_location='cpu')
    model.load_state_dict(initial,strict=True);model.requires_grad_(False)
    if family=='ann2bit':assert model.content_family=='ann2bit' and model.content_full_packet
    folder.mkdir(parents=True,exist_ok=False);(folder/'packets').mkdir()
    manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,config=str(config),config_sha256=sha(config),
        data_manifest_sha256=sha(cfg['data_manifest']),split_sha256={k:sha(cfg[k]) for k in ('root_dir','validate_dir')},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),
        family=family,mode=mode,limit=limit,seed=20260929,training=False,
        cuda_allocator_memory_fraction=.2,scale_policy='ANN transmits 32 FP32 scales; IF unit scales declared',
        comparison='Frozen historical systems; different receivers, selection and training; not causal activation-only control')
    write_json(folder/'manifest_snapshot.json',manifest)
    wrapper=FrozenSnapshot(model,family,mode,folder,original_check=limit==2).eval()
    result=validate(wrapper,dataset,2,limit=limit,include_3d=True,prediction_dir=folder/'predictions',profile=False)
    assert result['samples']==len(wrapper.records)==(limit or 1738)
    assert all(torch.equal(value.cpu(),initial[k]) for k,value in model.state_dict().items()) and sha(checkpoint)==digest
    index=read(folder/'predictions/index.json')
    for frame,record in zip(index['frames'],wrapper.records):
        assert frame['frame_id']==record['frame_id'] and frame['communication']['total_bytes']==record['actual_bytes']
        path=folder/'packets'/record['packet_file'];assert sha(path)==record['packet_sha256'] and path.stat().st_size==record['actual_bytes']
        if mode=='b2048':assert record['actual_bytes']<=2048
    write_json(folder/'wire_audit.json',dict(passed=True,all_state_unchanged=True,records=wrapper.records,
        timing_scope='includes selection, actual codec, duplicate direct/wire readout and audit; not deployment latency'))
    result['wire_audit_sha256']=sha(folder/'wire_audit.json');write_json(folder/'evaluation.json',result)
    print(json.dumps(dict(family=family,mode=mode,frames=result['samples'],ap=result['ap'],bytes=result['feature_packets'])),flush=True)


def coordinate(root,gpu):
    from opencood.tools.launch_e3dsnn_joint_round import check_gpu
    try:
        for stage,limit in (('smoke',2),('audit64',64),('full',0)):
            for family in SOURCES:
                for mode in MODES:
                    name=family+'_'+mode;folder=root/stage/name
                    write_json(root/'status.json',dict(state='running',stage=stage,current=name))
                    write_json(root/(stage+'_'+name+'_admission.json'),check_gpu(gpu,True,.2))
                    with (root/(stage+'_'+name+'.log')).open('xb') as log:
                        subprocess.run([sys.executable,'-u','-m','opencood.tools.evaluate_e3dsnn_ann_snapshot',
                            '--root',str(folder),'--family',family,'--mode',mode,'--limit',str(limit)],
                            cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
                    with (root/(stage+'_'+name+'_replay.log')).open('xb') as log:
                        subprocess.run([sys.executable,'-u','-m','opencood.tools.replay_e3dsnn_predictions','--evaluation-dir',str(folder)],
                            cwd=str(REPO),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
                    replay=read(folder/'offline_replay.json');assert replay['passed'] and replay['exact_match'] and replay['all_file_hashes_verified']
                    audit=read(folder/'wire_audit.json');assert audit['passed'] and audit['all_state_unchanged']
                    if stage=='smoke' and mode=='full':assert all(r['original_full_exact'] for r in audit['records'])
                    if stage=='audit64':
                        mean=float(np.mean([r['audit_forward_seconds'] for r in audit['records']]))
                        assert mean<=5.,'Reference forward audit exceeds pre-registered execution cost gate'
            write_json(root/(stage+'_complete.json'),dict(passed=True,limit=limit))
        results={}
        for family in SOURCES:
            for mode in MODES:
                name=family+'_'+mode;folder=root/'full'/name;row=evaluation(folder)
                assert row['manifest']['checkpoint_sha256']==SOURCES[family][2]
                audit=read(folder/'wire_audit.json')
                assert row['evaluation']['wire_audit_sha256']==sha(folder/'wire_audit.json')
                for record in audit['records']:
                    p=folder/'packets'/record['packet_file'];assert sha(p)==record['packet_sha256'] and p.stat().st_size==record['actual_bytes']
                results[name]=row
        write_json(root/'summary.json',dict(passed=True,results=results,protocol=read(root/'protocol.json')))
        write_json(root/'status.json',dict(state='complete',summary_sha256=sha(root/'summary.json')))
    except Exception:
        write_json(root/'status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry'))
        raise


def launch(root,gpu):
    from opencood.tools.launch_e3dsnn_joint_round import check_gpu
    if root.exists():raise FileExistsError('Fresh pipeline root required')
    for checkpoint,_,digest in SOURCES.values():assert sha(checkpoint)==digest
    admission=check_gpu(gpu,True,.2);root.mkdir()
    write_json(root/'protocol.json',dict(sources={f:dict(checkpoint=str(c),config=str(y),sha256=h) for f,(c,y,h) in SOURCES.items()},
        modes=list(MODES),budget_bytes=2048,seed=20260929,training=False,
        gates='smoke2 original full source equality and CPU replay; audit64 wire equality, hard bytes and <=5s audit forward; full1738',
        ann_selection='original raw-feature L1 ordering within original 50% candidates; complete 2bit rows; first actual-byte overflow stops',
        snn_selection='original IF unit-event ordering; first actual-byte overflow stops',
        codec='common E3TS1 spatial rows/channel mask; nonzero values 2bit ANN or 4bit IF; ANN FP32 channel scales included',
        limitation='different historical training, receivers and selection; engineering reference, not isolated SNN causality'))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/'pipeline.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.evaluate_e3dsnn_ann_snapshot','--root',str(root),'--coordinate','--gpu',str(gpu)],
            cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,gpu=gpu,admission=admission,started_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip())
    write_json(root/'launch.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--family',choices=tuple(SOURCES));p.add_argument('--mode',choices=MODES)
    p.add_argument('--limit',type=int,choices=(0,2,64),default=0);p.add_argument('--coordinate',action='store_true')
    p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    if a.family:execute(Path(a.root),a.family,a.mode,a.limit)
    elif a.coordinate:coordinate(Path(a.root),a.gpu)
    else:launch(Path(a.root),a.gpu)
