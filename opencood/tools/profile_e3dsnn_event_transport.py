"""Fixed-content E1a wire-cost study, validation prefix only, no AP claim."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.launch_e3dsnn_hybrid_round import sha
from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire, stream_readout
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.utils import event_stream_packet as raw
from opencood.utils import event_stream_compact as compact
from opencood.utils.e3dsnn_transplant import encoder_digest


def run(root, base, frames):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);(root/'cache').mkdir()
    base=Path(base)
    source=base/'e3dsnn_joint_round1_20261019/joint_adapt10/best_ap70.pth'
    assert sha(source)=='f4a5795b8b001a7cc4c04fd9eb3959f02df70181afe23691e733cb3aa4eb8588'
    config=base/'e3dsnn_event_e0_repack_20261020/config.yaml'
    cfg=load_yaml(str(config))
    random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929)
    model=train_utils.create_model(cfg).cuda().eval()
    model.load_state_dict(torch.load(str(source),map_location='cpu'),strict=True)
    encoder=encoder_digest(model);basis=model.event_basis_digest()
    dataset=build_dataset(cfg,train=False)
    if frames>len(dataset):raise ValueError('Requested more than available frames')
    report=dict(protocol='E1a fixed IF content; no training; not AP evaluation',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        checkpoint_sha256=sha(source),config_sha256=sha(config),frames=frames,
        validation_indices=list(range(frames)),records=[],budgets=[],prediction_checks=[])
    with torch.no_grad():
        for index in range(frames):
            batch=train_utils.to_device(dataset.collate_batch_test([dataset[index]]),torch.device('cuda'))['ego']
            ego,membrane=model.encode_membrane(batch['processed_lidar'],1)
            road=model.encode_lidar(batch['road_lidar'],1).dense()
            stream=integrate_and_fire(torch.relu(model.content.project(road)),model.event_max_spikes)
            c=MessageContract(str(batch['frame_ids'][0]),str(batch['road_frame_ids'][0]),
                'infrastructure',encoder,'event-content32-'+basis,tuple(int(n) for n in membrane.shape[2:]),
                tuple(model.origin),tuple(model.step),tuple(int(n) for n in batch['timestamps_us'][0].cpu().tolist()))
            pose=batch['road_to_ego']
            meta=dict(contract=asdict(c),source_to_ego=pose[0].cpu().tolist(),channels=32,
                      max_spikes=model.event_max_spikes,schema='if-unit-events-v1')
            np.savez_compressed(str(root/'cache'/('{:04d}.npz'.format(index))),
                addresses=stream.addresses,metadata=json.dumps(meta,sort_keys=True))
            frame_hash=hashlib.sha256(stream.addresses.tobytes()).hexdigest()
            for limit in (512,2048):
                ids=stream.addresses[:limit]
                expected=np.bincount(ids.astype(np.int64),minlength=stream.drive.numel()).reshape(stream.drive.shape)
                prediction=None
                for capacity in (64,256,1024):
                    for mode in ('raw_v1','session_raw','session_zlib'):
                        begin=time.perf_counter()
                        if mode=='raw_v1':
                            packets=[raw.encode(ids[i:i+capacity],meta,i) for i in range(0,len(ids),capacity)]
                            acc=raw.StreamAccumulator(meta)
                        else:
                            packets=compact.pack_all(ids,meta,capacity,mode=='session_zlib')
                            acc=compact.CompactAccumulator(meta)
                        encoded=time.perf_counter()
                        recovered=[]
                        for p in packets:
                            recovered.append(raw.decode(p)[0] if mode=='raw_v1' else compact.decode(p,meta)[0])
                            acc.receive(p)
                        counts=acc.finish()
                        assert np.array_equal(counts,expected)
                        assert np.array_equal(np.concatenate(recovered) if recovered else ids[:0],ids)
                        done=time.perf_counter()
                        report['records'].append(dict(index=index,frame=c.receiver_frame,limit=limit,events=len(ids),
                            stream_sha256=frame_hash,capacity=capacity,mode=mode,packets=len(packets),
                            bytes=sum(map(len,packets)),encode_ms=1000*(encoded-begin),decode_ms=1000*(done-encoded)))
                        # Two same-event transports per prefix check actual detector too.
                        if capacity==256 and mode in ('raw_v1','session_zlib'):
                            feature=stream_readout(membrane,road.new_tensor(counts),pose,c,
                                                   model.joint_receiver,model.content.decode)
                            detected=model.detect_bev(feature.flatten(1,2))
                            if prediction is not None:
                                assert all(torch.equal(prediction[k],detected[k]) for k in ('psm','rm','dm'))
                            prediction=detected
                report['prediction_checks'].append(dict(index=index,limit=limit,exact=True))
            for budget in (2048,4096,8192):
                for mode in ('raw_v1','session_zlib'):
                    fn=raw.pack_prefix if mode=='raw_v1' else compact.pack_prefix
                    packets,n,used=fn(stream.addresses,meta,budget,256)
                    acc=raw.StreamAccumulator(meta) if mode=='raw_v1' else compact.CompactAccumulator(meta)
                    for p in packets:acc.receive(p)
                    assert np.array_equal(acc.finish(),stream.prefix(n).cpu().numpy())
                    assert used==sum(map(len,packets)) and used<=budget
                    report['budgets'].append(dict(index=index,mode=mode,budget=budget,bytes=used,events=n))
            (root/'progress.json').write_text(json.dumps(dict(completed_frames=index+1,total=frames)))
    grouped={}
    for r in report['records']:
        key='{}:{}:{}'.format(r['limit'],r['capacity'],r['mode'])
        grouped.setdefault(key,[]).append(r)
    report['summary']={k:dict(frames=len(v),mean_bytes=float(np.mean([x['bytes'] for x in v])),
        p95_bytes=float(np.percentile([x['bytes'] for x in v],95)),
        mean_encode_ms=float(np.mean([x['encode_ms'] for x in v])),
        mean_decode_ms=float(np.mean([x['decode_ms'] for x in v]))) for k,v in grouped.items()}
    report['passed']=True
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=True,frames=frames,summary=report['summary']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--base',default='/data1/tzm/workspace/diagnostics')
    parser.add_argument('--frames',type=int,default=64)
    args=parser.parse_args()
    if not 1<=args.frames<=64:raise ValueError('Registered pilot is at most 64 frames')
    torch.cuda.set_per_process_memory_fraction(.2)
    run(args.root,args.base,args.frames)
