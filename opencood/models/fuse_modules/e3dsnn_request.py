"""Frame-local scalar spatial requests; no cross-channel feature addition."""
import json
import struct
import zlib
import numpy as np
import torch
import torch.nn.functional as F
from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume
from opencood.utils import task_snapshot_packet as task
from opencood.utils.quantized_feature_packet import pack,unpack

HEADER=struct.Struct('<5sIII')
POLICY='positive-count4-residual-max-2bit-xy4-v1'


def request_codes(membrane):
    if membrane.ndim!=5 or membrane.shape[0]!=1 or not torch.isfinite(membrane).all():raise ValueError('Expected one finite membrane frame')
    u=membrane.clamp(0,4);q=torch.floor(u+.5)
    score=(2*(u-q).clamp(min=0)).amax(1,keepdim=True)
    # Retain every height plane; pool XY blocks only, including cropped edges.
    coarse=F.max_pool3d(score,kernel_size=(1,4,4),stride=(1,4,4),ceil_mode=True)
    return torch.floor(coarse*3+.5)[0,0].to(torch.uint8).cpu().numpy()


def _context(meta):
    c=task.validate_metadata(meta)
    context=json.dumps(dict(task=meta,request=POLICY),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    shape=(c.shape_zyx[0],(c.shape_zyx[1]+3)//4,(c.shape_zyx[2]+3)//4)
    return c,context,shape


def encode_request(codes,meta):
    _,context,shape=_context(meta);a=np.asarray(codes)
    if a.shape!=shape or not np.isfinite(a).all() or np.any(a!=np.floor(a)) or np.any(a<0) or np.any(a>3):raise ValueError('Invalid spatial request codes')
    if not a.any():return b''  # Explicit no-request fallback, not a zero observation.
    raw=pack(a.ravel().astype(np.uint8),2).tobytes();body=context+zlib.compress(raw)
    return HEADER.pack(b'E3RQ1',len(context),len(raw),zlib.crc32(body))+body


def decode_request(packet,meta):
    c,context,shape=_context(meta)
    if not packet:return np.zeros(shape,dtype=np.uint8)
    if len(packet)<HEADER.size:raise ValueError('Truncated request')
    magic,n,rsize,crc=HEADER.unpack_from(packet);size=int(np.prod(shape));body=packet[HEADER.size:]
    if magic!=b'E3RQ1' or n!=len(context) or rsize!=(2*size+7)//8 or body[:n]!=context or zlib.crc32(body)!=crc:raise ValueError('Changed request identity or corrupt payload')
    decoder=zlib.decompressobj();raw=decoder.decompress(body[n:],rsize+1)
    if len(raw)!=rsize or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:raise ValueError('Invalid request compression')
    codes=unpack(np.frombuffer(raw,dtype=np.uint8),size,2).reshape(shape)
    # Reject noncanonical padding and serialized all-zero requests.
    if encode_request(codes,meta)!=packet:raise ValueError('Noncanonical request')
    return codes


def request_on_sender(codes,meta,device):
    c,_,shape=_context(meta)
    if np.asarray(codes).shape!=shape:raise ValueError('Wrong request shape')
    native=np.repeat(np.repeat(codes,4,axis=1),4,axis=2)[:,:c.shape_zyx[1],:c.shape_zyx[2]]
    volume=torch.as_tensor(native.copy(),device=device,dtype=torch.float32)[None,None]/3
    road_to_ego=torch.tensor(meta['source_to_ego'],device=device,dtype=torch.float64)[None]
    # warp_volume pulls its source onto target: source is now ego, target road.
    result,coverage=warp_volume(volume,torch.inverse(road_to_ego),c.origin_xyz,c.step_xyz,geometry_dtype=torch.float64)
    # Float32 grid_sample can leave tiny positive values exactly outside its
    # zero-padded domain; those must not win priority ties against true zeros.
    z,y,x=torch.meshgrid(*(torch.arange(n,device=device,dtype=torch.float64) for n in c.shape_zyx))
    origin=torch.tensor(c.origin_xyz,device=device,dtype=torch.float64);step=torch.tensor(c.step_xyz,device=device,dtype=torch.float64)
    xyz=torch.stack((x,y,z),-1)*step+origin
    target=xyz@road_to_ego[0,:3,:3].T+road_to_ego[0,:3,3]
    index=(target-origin)/step
    size=torch.tensor(c.shape_zyx[::-1],device=device,dtype=torch.float64)
    support=((index>-1)&(index<size)).all(-1)[None,None]
    return torch.where(support,result,torch.zeros_like(result)),torch.where(support,coverage,torch.zeros_like(coverage))


def request_order(addresses,times,request):
    ids=np.asarray(addresses);t=np.asarray(times);q=np.asarray(request).ravel()
    if ids.ndim!=1 or t.shape!=ids.shape or not np.isfinite(q).all() or np.any(q<0) or np.any(q>1+1e-6):raise ValueError('Invalid event request priorities')
    if not len(q) or np.any(ids<0) or np.any(ids>=32*len(q)):raise ValueError('Address outside task grid')
    # Requests describe places, not feature-channel values. All channels sample
    # the same spatial request; original IF time and address resolve ties.
    order=np.lexsort((ids,t,-q[ids%len(q)]))
    return ids[order]
