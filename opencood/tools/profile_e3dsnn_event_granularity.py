"""Fixed-prefix snapshot cost diagnostic, NOT an online incremental protocol."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import zlib
import numpy as np
from opencood.utils.quantized_feature_packet import pack, unpack
from opencood.utils.event_stream_packet import validate_metadata

HEADER = struct.Struct('<5sIIIII')


def encode_snapshot(counts, metadata):
    contract = validate_metadata(metadata)
    counts = np.asarray(counts)
    shape = (1,32)+contract.shape_zyx
    if (metadata['max_spikes'] > 15 or counts.shape != shape or
            not np.isfinite(counts).all() or np.any(counts != np.floor(counts)) or
            np.any(counts < 0) or np.any(counts > metadata['max_spikes'])):
        raise ValueError('Invalid snapshot counts')
    rows = counts.reshape(32,-1).T
    coords = np.flatnonzero(np.any(rows != 0,axis=1)).astype('<u4')
    if len(coords)==0: return b''
    rows=rows[coords];mask=rows!=0;values=rows[mask]
    raw=coords.tobytes()+pack(mask,1).tobytes()+pack(values,4).tobytes()
    context=json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
    body=context+zlib.compress(raw)
    return HEADER.pack(b'E3GS1',len(context),len(coords),len(values),len(raw),zlib.crc32(body))+body


def decode_snapshot(packet, metadata):
    contract=validate_metadata(metadata)
    if metadata['max_spikes']>15:raise ValueError('Unsupported snapshot ceiling')
    shape=(1,32)+contract.shape_zyx
    if not packet:return np.zeros(shape,dtype=np.uint8)
    if len(packet)<HEADER.size:raise ValueError('Truncated snapshot')
    magic,msize,n,nnz,rsize,crc=HEADER.unpack(packet[:HEADER.size])
    if (magic!=b'E3GS1' or not 1<=n<=np.prod(contract.shape_zyx) or not n<=nnz<=32*n or
            rsize!=8*n+(nnz+1)//2 or msize>65536 or len(packet)<=HEADER.size+msize):
        raise ValueError('Invalid snapshot header')
    context=json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
    body=packet[HEADER.size:]
    if body[:msize]!=context or zlib.crc32(body)!=crc:raise ValueError('Changed session or corrupt payload')
    decoder=zlib.decompressobj();raw=decoder.decompress(body[msize:],rsize+1)
    if len(raw)!=rsize or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('Wrong decompressed length')
    coords=np.frombuffer(raw[:4*n],dtype='<u4')
    if np.any(coords>=np.prod(contract.shape_zyx)) or np.any(coords[1:]<=coords[:-1]):
        raise ValueError('Invalid spatial rows')
    mask=unpack(np.frombuffer(raw[4*n:8*n],dtype=np.uint8),32*n,1).reshape(n,32).astype(bool)
    values=unpack(np.frombuffer(raw[8*n:],dtype=np.uint8),nnz,4)
    if int(mask.sum())!=nnz or not np.all(mask.any(1)) or np.any(values==0) or np.any(values>metadata['max_spikes']):
        raise ValueError('Invalid channel values')
    selected=np.zeros((n,32),dtype=np.uint8);selected[mask]=values
    rows=np.zeros((int(np.prod(contract.shape_zyx)),32),dtype=np.uint8);rows[coords]=selected
    return rows.T.reshape(shape)


def run(root):
    import random,time,subprocess,torch
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.tools.launch_e3dsnn_hybrid_round import sha
    from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire,stream_readout
    from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
    from opencood.utils import event_stream_compact as compact
    from opencood.utils.e3dsnn_transplant import encoder_digest
    root=Path(root);root.mkdir(parents=True,exist_ok=False);(root/'cache').mkdir()
    source=root.parent/'e3dsnn_event_receiver_20261021/receiver'
    checkpoint=source/'train/best_ap70.pth';config=source/'b4096.yaml'
    assert sha(checkpoint)=='60b7e0ff84e992862902aa9325b56ac13ab5658d70c4aacb194028638f5105dc'
    random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929)
    cfg=load_yaml(str(config));model=train_utils.create_model(cfg).cuda().eval()
    model.load_state_dict(torch.load(str(checkpoint),map_location='cpu'),strict=True)
    dataset=build_dataset(cfg,train=False);encoder=encoder_digest(model);basis=model.event_basis_digest()
    report=dict(started_unix=time.time(),checkpoint_sha256=sha(checkpoint),config_sha256=sha(config),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),frames=64,records=[],
        protocol='Fixed received prefix counts; one snapshot cannot be repeatedly added; no AP or training')
    with torch.no_grad():
        for index in range(64):
            data=train_utils.to_device(dataset.collate_batch_test([dataset[index]]),torch.device('cuda'))['ego']
            ego,membrane=model.encode_membrane(data['processed_lidar'],1)
            road=model.encode_lidar(data['road_lidar'],1).dense()
            drive=torch.relu(model.content.project(road));stream=integrate_and_fire(drive,model.event_max_spikes)
            c=MessageContract(str(data['frame_ids'][0]),str(data['road_frame_ids'][0]),'infrastructure',encoder,
                'event-content32-'+basis,tuple(int(x) for x in membrane.shape[2:]),tuple(model.origin),tuple(model.step),
                tuple(int(x) for x in data['timestamps_us'][0].cpu().tolist()))
            pose=data['road_to_ego'];meta=dict(contract=asdict(c),source_to_ego=pose[0].cpu().tolist(),channels=32,
                max_spikes=model.event_max_spikes,schema='if-unit-events-v1')
            np.savez_compressed(str(root/'cache'/('{:04d}.npz'.format(index))),addresses=stream.addresses,
                metadata=json.dumps(meta,sort_keys=True))
            for budget in (4096,8192,16384):
                packets,n,used=compact.pack_prefix(stream.addresses,meta,budget,256)
                counts=stream.prefix(n);array=counts.cpu().numpy();ids=stream.addresses[:n]
                snapshot=encode_snapshot(array,meta);restored=decode_snapshot(snapshot,meta)
                assert np.array_equal(restored,array)
                one=compact.encode(ids,meta,0) if n else b''
                if n:assert np.array_equal(compact.decode(one,meta)[0],ids)
                a=stream_readout(membrane,counts,pose,c,model.joint_receiver,model.content.decode)
                b=stream_readout(membrane,counts.new_tensor(restored),pose,c,model.joint_receiver,model.content.decode)
                assert torch.equal(a,b)
                pa,pb=model.detect_bev(a.flatten(1,2)),model.detect_bev(b.flatten(1,2))
                assert all(torch.equal(pa[k],pb[k]) for k in ('psm','rm','dm'))
                spatial=array.reshape(32,-1).T;occupied=spatial[np.any(spatial!=0,axis=1)]
                report['records'].append(dict(index=index,frame=c.receiver_frame,budget=budget,events=n,
                    unique_channel_addresses=int(np.count_nonzero(array)),unique_voxels=len(occupied),
                    mean_active_channels_per_voxel=float(np.mean(np.count_nonzero(occupied,axis=1))) if len(occupied) else 0.,
                    count_histogram=np.bincount(array[array>0].astype(int),minlength=9).tolist(),
                    deployed_bytes=used,single_event_packet_bytes=len(one),spatial_snapshot_bytes=len(snapshot),
                    counts_state_predictions_exact=True))
            if (index+1)%8==0:print(json.dumps(dict(frames=index+1)),flush=True)
    report['summary']={str(b):{k:float(np.mean([r[k] for r in report['records'] if r['budget']==b]))
        for k in ('events','unique_channel_addresses','unique_voxels','mean_active_channels_per_voxel',
                  'deployed_bytes','single_event_packet_bytes','spatial_snapshot_bytes')} for b in (4096,8192,16384)}
    report['passed']=True;report['finished_unix']=time.time()
    (root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    args=parser.parse_args()
    import torch
    torch.cuda.set_per_process_memory_fraction(.2)
    run(args.root)
