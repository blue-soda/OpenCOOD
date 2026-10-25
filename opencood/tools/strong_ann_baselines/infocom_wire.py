"""Exact FP32 InfoCom latent + spatial-mask transport, independent of Torch.

This is our measured reference codec, not a codec supplied by the authors.
Sparse mode carries a bitmap, exact FP32-value dictionary and packed codes.
Dense mode carries FP32 mask values. Choose the smaller zlib stream including
the same fixed header. A separate optional 80-byte envelope carries frame,
sender, sequence and 4x4 FP32 transform; report it separately for fair accounting.
"""
import struct
import zlib
import time
import hashlib
from pathlib import Path
import numpy as np

HEADER = struct.Struct('<4sBBHIIIII')
SPARSE = struct.Struct('<III')
ENVELOPE = struct.Struct('<QII16f')


def encode(latent, mask):
    latent = np.asarray(latent, dtype='<f4').reshape(-1)
    mask = np.asarray(mask, dtype='<f4')
    if mask.ndim != 2 or not np.isfinite(mask).all() or not np.isfinite(latent).all():
        raise ValueError('Expected finite vector and HxW mask')
    flat = mask.reshape(-1)
    # Compare uint32 to preserve negative-zero bit patterns as well.
    words = flat.view('<u4'); selected = words != 0
    dictionary, inverse = np.unique(words[selected], return_inverse=True)
    bits = max(1, int(max(0, len(dictionary)-1)).bit_length())
    codes = ((inverse.astype(np.uint64)[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8)
    bitmap = np.packbits(selected.astype(np.uint8)).tobytes()
    sparse = (latent.tobytes() + SPARSE.pack(int(selected.sum()), len(dictionary), bits)
              + bitmap + dictionary.astype('<u4').tobytes() + np.packbits(codes.reshape(-1)).tobytes())
    dense = latent.tobytes() + flat.tobytes()
    options = [(zlib.compress(dense, 6), 0, len(dense)), (zlib.compress(sparse, 6), 1, len(sparse))]
    payload, mode, raw_len = min(options, key=lambda x: len(x[0]))
    packet = HEADER.pack(b'ICW1', 1, mode, 0, mask.shape[0], mask.shape[1], len(latent), len(payload), raw_len) + payload
    return packet, {'mode': mode, 'nonzero_values': int(selected.sum()), 'dictionary_values': len(dictionary),
                    'code_bits': bits, 'packet_bytes': len(packet), 'header_bytes': HEADER.size,
                    'compressed_body_bytes': len(payload), 'uncompressed_body_bytes': raw_len,
                    'dense_compressed_bytes': len(options[0][0])+HEADER.size,
                    'sparse_compressed_bytes': len(options[1][0])+HEADER.size,
                    'paper_value_only_bytes': len(latent)*4 + int(np.ceil(selected.sum()/2.))}


def decode(packet):
    if len(packet)<HEADER.size:raise ValueError('Truncated header')
    magic,version,mode,reserved,h,w,dim,size,raw_len = HEADER.unpack_from(packet)
    if magic!=b'ICW1' or version!=1 or reserved or mode not in (0,1):raise ValueError('Unsupported packet')
    if not 0<h*w<=10000000 or not 0<dim<=65536 or size!=len(packet)-HEADER.size or raw_len>100000000:
        raise ValueError('Invalid dimensions/length')
    raw=zlib.decompress(packet[HEADER.size:])
    if len(raw)!=raw_len:raise ValueError('Invalid uncompressed length')
    latent=np.frombuffer(raw[:dim*4],dtype='<f4').copy();offset=dim*4
    if mode==0:
        if len(raw)!=offset+h*w*4:raise ValueError('Invalid dense length')
        mask=np.frombuffer(raw[offset:],dtype='<f4').copy().reshape(h,w)
    else:
        n,unique,bits=SPARSE.unpack_from(raw,offset);offset+=SPARSE.size
        if bits!=max(1,int(max(0,unique-1)).bit_length()) or n>h*w or unique>n:raise ValueError('Invalid sparse layout')
        nbitmap=(h*w+7)//8
        selected=np.unpackbits(np.frombuffer(raw[offset:offset+nbitmap],dtype=np.uint8))[:h*w].astype(bool);offset+=nbitmap
        if selected.sum()!=n:raise ValueError('Bitmap count mismatch')
        dictionary=np.frombuffer(raw[offset:offset+unique*4],dtype='<u4');offset+=unique*4
        if len(raw)!=offset+(n*bits+7)//8:raise ValueError('Invalid codes length')
        codebits=np.unpackbits(np.frombuffer(raw[offset:],dtype=np.uint8))[:n*bits].reshape(n,bits)
        codes=(codebits.astype(np.uint64) << np.arange(bits,dtype=np.uint64)).sum(axis=1).astype(np.int64)
        if n and (not unique or codes.max()>=unique):raise ValueError('Code outside dictionary')
        words=np.zeros(h*w,dtype='<u4');words[selected]=dictionary[codes]
        mask=words.view('<f4').reshape(h,w)
    return latent,mask


def envelope(frame_id,sender,transform,sequence=0):
    transform=np.asarray(transform,dtype=np.float32)
    if transform.shape!=(4,4) or not np.isfinite(transform).all():raise ValueError('Invalid pose')
    return ENVELOPE.pack(int(frame_id),int(sender),int(sequence),*transform.reshape(-1))


class WireAudit:
    """Substitute decoded messages into the real decoder; ego remains local."""
    def __init__(self,model,output):
        self.output=Path(output)/'wire';self.output.mkdir()
        self.handle=model.ib_communication.decoder.register_forward_pre_hook(self.hook)
        self.messages=[]

    def begin(self,batch,frame_id):
        self.batch=batch;self.frame_id=frame_id;self.messages=[]

    def hook(self,module,inputs):
        import torch
        latent,mask=inputs
        n=self.batch['ego']['record_len'].detach().cpu().tolist()
        if len(n)!=1 or n[0]!=latent.shape[0] or mask.shape[:2]!=(n[0],1):
            raise ValueError('Wire audit requires one sample, ego first')
        replacement_l=latent.clone();replacement_m=mask.clone()
        for sender in range(1,n[0]):
            start=time.perf_counter()
            l=latent[sender].detach().cpu().numpy();m=mask[sender,0].detach().cpu().numpy()
            if l.dtype!=np.float32 or m.dtype!=np.float32:raise ValueError('FP32 required')
            copied=time.perf_counter();packet,record=encode(l,m);encoded=time.perf_counter()
            dl,dm=decode(packet);decoded=time.perf_counter()
            if dl.tobytes()!=l.tobytes() or dm.tobytes()!=m.tobytes():raise ValueError('Non-exact round trip')
            replacement_l[sender]=torch.from_numpy(dl).to(latent.device)
            replacement_m[sender,0]=torch.from_numpy(dm).to(mask.device)
            # Convention is the exact official pairwise_t_matrix[0,0,sender].
            transform=self.batch['ego']['pairwise_t_matrix'][0,0,sender].detach().cpu().numpy()
            meta=envelope(self.frame_id,sender,transform)
            record.update(sender=sender,frame_id=str(self.frame_id),envelope_bytes=len(meta),
                          total_bytes=len(meta)+len(packet),sha256=hashlib.sha256(meta+packet).hexdigest(),
                          host_copy_seconds=copied-start,encode_seconds=encoded-copied,decode_seconds=decoded-encoded,
                          roundtrip_seconds=time.perf_counter()-start,exact_fp32=True)
            self.messages.append((record,meta+packet))
        return replacement_l,replacement_m

    def finish(self,index):
        rows=[]
        for record,packet in self.messages:
            p=self.output/('%06d_sender%d.bin'%(index,record['sender']));p.write_bytes(packet)
            record['file']='wire/'+p.name;rows.append(record)
        return {'messages':rows,'packet_bytes':sum(x['packet_bytes'] for x in rows),
                'envelope_bytes':sum(x['envelope_bytes'] for x in rows),
                'total_bytes':sum(x['total_bytes'] for x in rows),
                'encode_seconds':sum(x['encode_seconds'] for x in rows),
                'decode_seconds':sum(x['decode_seconds'] for x in rows)}


def self_test():
    rng=np.random.RandomState(303)
    for shape in [(1,1),(3,5),(200,504)]:
        latent=rng.randn(256).astype(np.float32)
        masks=[np.zeros(shape,np.float32),rng.randn(*shape).astype(np.float32),
               (rng.randint(0,16,size=shape).astype(np.float32)/15)*(rng.rand(*shape)<.1)]
        masks[-1].flat[0]=-0.
        for mask in masks:
            wire,_=encode(latent,mask);got_l,got_m=decode(wire)
            assert got_l.tobytes()==latent.tobytes() and got_m.tobytes()==mask.tobytes()
            try:decode(wire[:-1])
            except (ValueError,zlib.error):pass
            else:raise AssertionError('Truncated packet accepted')
    assert len(envelope(211,1,np.eye(4)))==80
    print('9 exact FP32 round trips, zero/negative-zero/dense/sparse and truncation checks passed')

if __name__=='__main__':self_test()
