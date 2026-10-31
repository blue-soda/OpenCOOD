"""Common spatial snapshot envelope for IF counts and scaled ANN integer codes.

ANN FP32 channel scales are transmitted, never supplied by the receiver. SNN
unit scales are an explicit schema convention. Counts are not an event sequence.
"""
import json
import struct
import zlib
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract, check_transform
from opencood.utils.quantized_feature_packet import pack, unpack

HEADER = struct.Struct('<5sIIIII')
SPECS = {'if8': (4, 8, 0), 'ann2bit': (2, 3, 128)}


def validate_metadata(meta):
    if set(meta) != {'contract','source_to_ego','family','schema'} or meta['schema'] != 'task-count32-v1':
        raise ValueError('Incomplete task snapshot identity')
    if meta['family'] not in SPECS: raise ValueError('Unknown quantization family')
    raw=dict(meta['contract'])
    for key in ('shape_zyx','origin_xyz','step_xyz','timestamps_us'): raw[key]=tuple(raw[key])
    contract=MessageContract(**raw)
    basis=contract.channel_basis
    if not basis.startswith('task32-') or len(basis)!=71 or any(c not in '0123456789abcdef' for c in basis[7:]):
        raise ValueError('Invalid task basis digest')
    if 32*int(np.prod(contract.shape_zyx))>=2**32: raise ValueError('Grid exceeds address space')
    check_transform(torch.tensor(meta['source_to_ego'],dtype=torch.float64)[None])
    return contract


def _context(meta):
    return json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')


def _scales(scales,family):
    scales=np.asarray(scales)
    if scales.shape!=(32,) or scales.dtype!=np.float32 or not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('Expected 32 positive finite FP32 scales')
    if family=='if8' and not np.all(scales==1): raise ValueError('IF schema has unit scales')
    return scales.astype('<f4').tobytes() if family=='ann2bit' else b''


def _encode_rows(coords,rows,context,scale_bytes,bits):
    if not len(coords): return b''
    mask=rows!=0;values=rows[mask]
    raw=scale_bytes+coords.astype('<u4').tobytes()+pack(mask,1).tobytes()+pack(values,bits).tobytes()
    body=context+zlib.compress(raw)
    return HEADER.pack(b'E3TS1',len(context),len(coords),len(values),len(raw),zlib.crc32(body))+body


def prepare(counts,scales,meta):
    c=validate_metadata(meta);bits,ceiling,_=SPECS[meta['family']]
    counts=np.asarray(counts)
    if (counts.shape!=(1,32)+c.shape_zyx or not np.isfinite(counts).all() or
            np.any(counts!=np.floor(counts)) or np.any(counts<0) or np.any(counts>ceiling)):
        raise ValueError('Invalid integer source counts')
    scale_bytes=_scales(scales,meta['family'])
    rows=counts.reshape(32,-1).T.astype(np.uint8)
    coords=np.flatnonzero(np.any(rows!=0,axis=1))
    return c,coords,rows[coords],_context(meta),scale_bytes,bits


def encode(counts,scales,meta):
    _,coords,rows,context,scale_bytes,bits=prepare(counts,scales,meta)
    return _encode_rows(coords,rows,context,scale_bytes,bits)


def decode(packet,expected_meta):
    c=validate_metadata(expected_meta);bits,ceiling,sbytes=SPECS[expected_meta['family']]
    counts=np.zeros((1,32)+c.shape_zyx,dtype=np.uint8)
    # Silence means no remote contribution; it does not transmit arbitrary scales.
    if not packet:return counts,np.ones(32,dtype=np.float32)
    if len(packet)<HEADER.size:raise ValueError('Truncated snapshot')
    magic,msize,n,nnz,rsize,crc=HEADER.unpack(packet[:HEADER.size]);size=int(np.prod(c.shape_zyx))
    if (magic!=b'E3TS1' or not 1<=n<=size or not n<=nnz<=32*n or
            rsize!=sbytes+8*n+(bits*nnz+7)//8 or msize>65536 or len(packet)<=HEADER.size+msize):
        raise ValueError('Invalid snapshot header')
    body=packet[HEADER.size:]
    if body[:msize]!=_context(expected_meta) or zlib.crc32(body)!=crc:raise ValueError('Changed identity or corrupt payload')
    decoder=zlib.decompressobj();raw=decoder.decompress(body[msize:],rsize+1)
    if len(raw)!=rsize or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:raise ValueError('Invalid compression stream')
    scales=np.frombuffer(raw[:sbytes],dtype='<f4').copy() if sbytes else np.ones(32,dtype=np.float32)
    _scales(scales,expected_meta['family']);raw=raw[sbytes:]
    coords=np.frombuffer(raw[:4*n],dtype='<u4')
    if np.any(coords>=size) or np.any(coords[1:]<=coords[:-1]):raise ValueError('Invalid spatial rows')
    mask=unpack(np.frombuffer(raw[4*n:8*n],dtype=np.uint8),32*n,1).reshape(n,32).astype(bool)
    values=unpack(np.frombuffer(raw[8*n:],dtype=np.uint8),nnz,bits)
    if int(mask.sum())!=nnz or not np.all(mask.any(1)) or np.any(values==0) or np.any(values>ceiling):raise ValueError('Invalid channel codes')
    rows=np.zeros((n,32),dtype=np.uint8);rows[mask]=values
    counts.reshape(32,size)[:,coords]=rows.T
    return counts,scales


def select(counts,scales,meta,budget=None,addresses=None,row_order=None):
    """Full snapshot, or first-overflow prefix at actual serialized byte limit."""
    c,coords,rows,context,scale_bytes,bits=prepare(counts,scales,meta)
    size=int(np.prod(c.shape_zyx));full=rows.copy()
    if budget is not None and (type(budget) is not int or budget<0):raise ValueError('Invalid budget')
    if budget is None:
        packet=_encode_rows(coords,rows,context,scale_bytes,bits);accepted=len(coords);rejected=None
    else:
        if (addresses is None)==(row_order is None):raise ValueError('Exactly one source ordering required')
        if addresses is not None:
            ids=np.asarray(addresses)
            if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=32*size):raise ValueError('Invalid event addresses')
            expected=np.bincount(ids.astype(np.int64),minlength=32*size).reshape(np.asarray(counts).shape)
            if not np.array_equal(expected,counts):raise ValueError('Events disagree with full content')
            slots=np.searchsorted(coords,ids%size);channels=ids//size
        else:
            ids=np.asarray(row_order)
            if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not np.array_equal(np.sort(ids),coords):raise ValueError('Row order must cover each nonzero source row once')
            slots=np.searchsorted(coords,ids)
        rows=np.zeros_like(full);active=np.zeros(len(coords),dtype=bool)
        packet=b'';accepted=0;rejected=None
        native = None
        if addresses is not None and meta['family'] == 'if8':
            from opencood.utils.snapshot_prefix_native import search
            native = search(coords, slots, channels, HEADER.size + len(context), budget)
        if native is not None:
            accepted, rejected = native
            np.add.at(rows, (slots[:accepted], channels[:accepted]), 1)
            active = np.any(rows != 0, axis=1)
            packet = _encode_rows(coords[active], rows[active], context, scale_bytes, bits)
        for i,slot in (enumerate(slots) if native is None else ()):
            old=rows[slot].copy();was_active=active[slot]
            if addresses is not None:rows[slot,channels[i]]+=1
            else:rows[slot]=full[slot]
            active[slot]=True
            candidate=_encode_rows(coords[active],rows[active],context,scale_bytes,bits)
            if len(candidate)>budget:
                rows[slot]=old;active[slot]=was_active;rejected=len(candidate);break
            packet=candidate;accepted+=1
        assert len(packet)<=budget
    selected=np.zeros((1,32)+c.shape_zyx,dtype=np.uint8)
    selected.reshape(32,size)[:,coords]=rows.T
    assert encode(selected,scales,meta)==packet
    restored,wire_scales=decode(packet,meta)
    assert np.array_equal(restored,selected)
    if packet:assert np.array_equal(wire_scales,scales)
    return packet,selected,dict(accepted_units=int(accepted),actual_bytes=len(packet),
        budget_bytes=budget,first_rejected_bytes=rejected,
        selected_events=int(selected.sum()),selected_voxels=int(np.any(selected,axis=1).sum()))
