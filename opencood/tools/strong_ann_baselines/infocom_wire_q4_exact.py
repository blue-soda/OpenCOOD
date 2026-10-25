"""Lossless 4-bit predictor plus FP32 XOR corrections for InfoCom masks.

This does NOT discard STE rounding differences. Original FP32 values (including
negative zero) are restored exactly; select the shorter of this and ICW1.
No model, threshold, spatial mask or latent vector is changed.
"""
import struct
import zlib
import numpy as np
try:
    from . import infocom_wire as legacy
except ImportError:
    import infocom_wire as legacy

HEADER=struct.Struct('<4sIIIIII')


def encode(latent,mask):
    latent=np.asarray(latent,dtype='<f4').reshape(-1)
    mask=np.asarray(mask,dtype='<f4')
    old,record=legacy.encode(latent,mask)
    flat=mask.reshape(-1)
    codes=np.rint(np.clip(flat,0,1)*np.float32(15)).astype(np.uint8)
    packed=codes[::2].copy()
    packed[:len(codes)//2] |= codes[1::2]<<4
    canonical=(codes.astype(np.float32)/np.float32(15)).astype('<f4')
    correction=flat.view('<u4') ^ canonical.view('<u4')
    # Byte planes isolate predominantly-zero high bytes, without dropping them.
    planes=correction.astype('<u4').view(np.uint8).reshape(-1,4).T.copy().tobytes()
    raw=latent.tobytes()+packed.tobytes()+planes
    body=zlib.compress(raw,6)
    header=HEADER.pack(b'ICQ1',mask.shape[0],mask.shape[1],len(latent),len(body),len(raw),0)
    candidate=header[:-4]+struct.pack('<I',zlib.crc32(header+body))+body
    result=candidate if len(candidate)<len(old) else old
    return result,dict(format='ICQ1' if result is candidate else 'ICW1',
        packet_bytes=len(result),legacy_bytes=len(old),q4_exact_bytes=len(candidate),
        correction_values=int(np.count_nonzero(correction)),exact_fp32=True)


def decode(packet):
    if packet[:4]==b'ICW1':return legacy.decode(packet)
    if len(packet)<HEADER.size:raise ValueError('Truncated ICQ1 header')
    magic,h,w,dim,size,raw_len,crc=HEADER.unpack_from(packet)
    n=h*w
    if magic!=b'ICQ1' or not 0<n<=10000000 or not 0<dim<=65536:
        raise ValueError('Invalid ICQ1 dimensions')
    expected=4*dim+(n+1)//2+4*n
    if raw_len!=expected or size!=len(packet)-HEADER.size:
        raise ValueError('Invalid ICQ1 length')
    header=HEADER.pack(magic,h,w,dim,size,raw_len,0);body=packet[HEADER.size:]
    if zlib.crc32(header+body)!=crc:raise ValueError('ICQ1 corruption')
    decoder=zlib.decompressobj();raw=decoder.decompress(body,raw_len+1)
    if len(raw)!=raw_len or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('Invalid ICQ1 compression stream')
    latent=np.frombuffer(raw[:dim*4],dtype='<f4').copy()
    offset=dim*4;packed=np.frombuffer(raw[offset:offset+(n+1)//2],dtype=np.uint8)
    codes=np.empty(n,dtype=np.uint8);codes[::2]=packed&15;codes[1::2]=packed[:n//2]>>4
    if n%2 and packed[-1]>>4:raise ValueError('Nonzero padding')
    offset+=(n+1)//2
    xor=np.frombuffer(raw[offset:],dtype=np.uint8).reshape(4,n).T.copy().view('<u4').reshape(-1)
    canonical=(codes.astype(np.float32)/np.float32(15)).astype('<f4')
    mask=(canonical.view('<u4')^xor).view('<f4').reshape(h,w)
    if not np.isfinite(latent).all() or not np.isfinite(mask).all():raise ValueError('Nonfinite message')
    return latent,mask


def self_test():
    rng=np.random.RandomState(303);formats=set();cases=0
    for shape in ((1,1),(3,5),(200,504)):
        latent=rng.randn(256).astype(np.float32)
        quant=rng.randint(0,16,size=shape).astype(np.float32)/np.float32(15)
        quant*=rng.rand(*shape)<.1
        perturbed=np.nextafter(quant,np.float32(1));perturbed[quant==0]=0
        for mask in (np.zeros(shape,np.float32),quant,perturbed,rng.randn(*shape).astype(np.float32)):
            mask=mask.copy();mask.flat[0]=-0.
            packet,record=encode(latent,mask);a,b=decode(packet)
            assert a.tobytes()==latent.tobytes() and b.tobytes()==mask.tobytes()
            assert len(packet)<=record['legacy_bytes'];formats.add(record['format']);cases+=1
            for bad in (packet[:-1],packet+b'x'):
                try:decode(bad)
                except (ValueError,zlib.error,struct.error):pass
                else:raise AssertionError('Invalid length accepted')
            if packet[:4]==b'ICQ1':
                bad=bytearray(packet);bad[5]^=1
                try:decode(bytes(bad))
                except ValueError:pass
                else:raise AssertionError('Corrupt header accepted')
    assert formats=={'ICW1','ICQ1'},formats
    print('%d exact round trips; both layouts; odd size, negative zero, ULP corrections, corruption checked'%cases)


if __name__=='__main__':self_test()
