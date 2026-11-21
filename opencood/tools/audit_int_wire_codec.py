"""Serialize and round-trip existing events with two shared address formats."""
import struct

import numpy as np


def fixed_pack(values,width):
    shifts=np.arange(width,dtype=np.uint32)
    bits=((values.astype(np.uint32)[:,None] >> shifts)&1).astype(np.uint8)
    return np.packbits(bits.reshape(-1),bitorder='little').tobytes()


def fixed_unpack(data,count,width):
    bits=np.unpackbits(np.frombuffer(data,dtype=np.uint8),bitorder='little')[:count*width].reshape(count,width)
    return (bits.astype(np.uint32) << np.arange(width,dtype=np.uint32)).sum(1,dtype=np.uint32)


def delta_pack(keys):
    gaps=np.diff(keys,prepend=np.uint32(0))
    pieces=gaps[:,None] >> np.arange(0,28,7,dtype=np.uint32)
    used=pieces!=0
    used[:,0]=True
    return ((pieces&127) | ((pieces>127).astype(np.uint32)<<7))[used].astype(np.uint8).tobytes()


def delta_unpack(data):
    values=np.frombuffer(data,dtype=np.uint8)
    if not len(values):return np.empty(0,dtype=np.uint32)
    ends=np.flatnonzero(values<128)
    assert ends[-1]==len(values)-1
    starts=np.r_[0,ends[:-1]+1]
    widths=ends-starts+1
    assert widths.max()<=4
    shifts=(np.arange(len(values))-np.repeat(starts,widths)).astype(np.uint32)*7
    terms=(values&127).astype(np.uint32)<<shifts
    return np.add.reduceat(terms,starts,dtype=np.uint32).cumsum(dtype=np.uint32)


def wire_audit(ids, sent, read, levels, floating):
    ids=ids.detach().cpu().numpy()
    values=sent.detach().cpu().numpy()
    area=read.numel()
    if area!=25200 or len(ids) and ids.max()>=128*area:
        raise ValueError('Unexpected shallow grid shape')
    keys=(ids%area*128+ids//area).astype(np.uint32)
    order=keys.argsort()
    keys,values=keys[order],values[order]
    assert np.all(keys[1:]>keys[:-1])
    if floating:
        payload=values.astype('<f4').tobytes()
    else:
        table=np.asarray(levels)
        symbols=np.searchsorted(table,np.abs(values))+(values<0)*4
        payload=fixed_pack(symbols,3)
    sizes=[]
    for mode,addresses in enumerate((fixed_pack(keys,22),delta_pack(keys))):
        packet=struct.pack('<BI',mode,len(keys))+addresses+payload
        kind,count=struct.unpack_from('<BI',packet)
        payload_len=count*4 if floating else (3*count+7)//8
        end=len(packet)-payload_len
        addr_data=packet[5:end]
        decoded=fixed_unpack(addr_data,count,22) if kind==0 else delta_unpack(addr_data)
        assert np.array_equal(decoded,keys)
        if floating:
            restored=np.frombuffer(packet[end:],dtype='<f4')
        else:
            symbol=fixed_unpack(packet[end:],count,3)
            restored=table[symbol%4]*np.where(symbol>=4,-1,1)
        assert np.array_equal(restored,values)
        sizes.append(len(packet))
    return dict(fixed_bytes=sizes[0],delta_bytes=sizes[1],best_bytes=min(sizes),
                fp32_same_addresses=min(sizes)-len(payload)+4*len(keys),payload=len(payload))
