"""Optional Count4 wire codec: mask zeros, encode nonzero values 1..4 in 2 bits.

Legacy codec untouched. Choose actual shorter bytes between legacy and offset
candidate; no change to features, spatial selection, or trained activations.
"""
import io
import json
import numpy as np
from opencood.utils import quantized_feature_packet as legacy


def encode(coords,features,metadata,options):
    old=legacy.encode(coords,features,metadata,'count4',options)
    coords,codes,scales=legacy.prepare(coords,features,'count4',options)
    mask=codes!=0;stream=io.BytesIO()
    np.savez_compressed(stream,coords=coords,scales=scales,
        codes=legacy.pack(codes[mask]-1,2),mask=np.packbits(mask.reshape(-1),bitorder='little'),
        header=np.asarray(json.dumps(dict(schema=2,bits=2,offset=1,sparse=True,
            activation='count4',options=options,metadata=metadata),sort_keys=True,
            separators=(',',':'),allow_nan=False)))
    return min((old,stream.getvalue()),key=len)


def decode(packet,options):
    legacy.validate_options(options)
    with np.load(io.BytesIO(packet),allow_pickle=False) as data:
        header=json.loads(str(data['header'].item()))
        if header.get('schema')==1:return legacy.decode(packet,'count4',options)
        coords,scales=data['coords'].copy(),data['scales'].copy()
        payload,mask_payload=data['codes'].copy(),data['mask'].copy()
    if (header.get('schema')!=2 or header.get('bits')!=2 or header.get('offset')!=1 or
            header.get('sparse') is not True or header.get('activation')!='count4' or
            header.get('options')!=options or options['bits']!=3):
        raise ValueError('Compact Count4 contract mismatch')
    shape=np.asarray(header['metadata']['shape_zyx'])
    if (shape.shape!=(3,) or np.any(shape<=0) or np.prod(shape)>1000000 or
            coords.ndim!=2 or coords.shape[1]!=3 or coords.dtype!=np.dtype('<u2') or
            np.any(coords>=shape) or len(np.unique(coords,axis=0))!=len(coords) or
            scales.shape!=(128,) or scales.dtype!=np.dtype('<f4') or not np.all(scales==1)):
        raise ValueError('Invalid grid, coordinates or scales')
    mask=legacy.unpack(mask_payload,len(coords)*128,1).astype(bool)
    values=legacy.unpack(payload,int(mask.sum()),2)+1
    codes=np.zeros(len(coords)*128,dtype=np.uint8);codes[mask]=values
    codes=codes.reshape(-1,128)
    if np.any(~np.any(codes!=0,axis=1)):raise ValueError('Redundant zero rows')
    return coords,codes.astype(np.float32),header['metadata']
