"""32-channel Count3/ANN message wire format; no selection or requantization."""
import io
import json
import numpy as np
from opencood.utils.quantized_feature_packet import pack, unpack


def _validate(coords, codes, scales, metadata, family, kind):
    if family not in ('count3', 'ann2bit') or kind not in ('event', 'full'):
        raise ValueError('Unknown content message format')
    if (codes.shape != (len(coords), 32) or coords.shape != (len(codes), 3) or
            not np.isfinite(coords).all() or np.any(coords != np.floor(coords)) or
            np.any(coords < 0) or np.any(coords >= 65536) or
            len(np.unique(coords, axis=0)) != len(coords)):
        raise ValueError('Invalid content coordinates or channel count')
    maximum = 1 if kind == 'event' else 3
    if (not np.isfinite(codes).all() or np.any(codes != np.floor(codes)) or
            np.any(codes < 0) or np.any(codes > maximum) or
            np.any(~np.any(codes != 0, axis=1)) or scales.shape != (32,) or
            not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError('Invalid content values or scales')
    if family == 'count3' and not np.all(scales == 1):
        raise ValueError('Count3 uses unit scale')
    from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract, check_transform
    import torch
    expected = {'contract', 'basis_sha256', 'source_to_ego', 'packet_index', 'packet_count'}
    if set(metadata) != expected or metadata['packet_count'] != 3:
        raise ValueError('Incomplete content metadata')
    contract = dict(metadata['contract'])
    for key in ('shape_zyx', 'origin_xyz', 'step_xyz', 'timestamps_us'):
        contract[key] = tuple(contract[key])
    contract = MessageContract(**contract)
    basis = metadata['basis_sha256']
    if (not isinstance(basis, str) or len(basis) != 64 or
            any(c not in '0123456789abcdef' for c in basis) or
            contract.channel_basis != 'content32-' + basis):
        raise ValueError('Invalid latent channel basis')
    if np.prod(contract.shape_zyx) > 1000000 or np.any(coords >= contract.shape_zyx):
        raise ValueError('Address outside declared grid')
    index = metadata['packet_index']
    if type(index) is not int or index not in ((1, 2, 3) if kind == 'event' else (3,)):
        raise ValueError('Invalid content packet index')
    check_transform(torch.tensor(metadata['source_to_ego'], dtype=torch.float64)[None])


def encode(coords, codes, scales, metadata, family, kind='event'):
    coords, codes, scales = np.asarray(coords), np.asarray(codes), np.asarray(scales)
    _validate(coords, codes, scales, metadata, family, kind)
    coords, codes, scales = coords.astype('<u2'), codes.astype('uint8'), scales.astype('<f4')
    # Validate after narrowing too: under/overflow must not change the protocol.
    _validate(coords, codes, scales, metadata, family, kind)
    bits = 1 if kind == 'event' else 2
    mask = codes != 0
    candidates = []
    for sparse in (False, True):
        header = dict(schema=1, channels=32, family=family, kind=kind, sparse=sparse, bits=bits, metadata=metadata)
        stream = io.BytesIO()
        np.savez_compressed(stream, coords=coords, scales=scales,
            values=pack(codes[mask] if sparse else codes, bits),
            mask=pack(mask, 1) if sparse else np.empty(0, dtype=np.uint8),
            header=np.asarray(json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False)))
        candidates.append(stream.getvalue())
    return min(candidates, key=len)


def decode(packet, family, kind='event'):
    with np.load(io.BytesIO(packet), allow_pickle=False) as data:
        coords, scales, values, mask = (data[k].copy() for k in ('coords', 'scales', 'values', 'mask'))
        h = json.loads(str(data['header'].item()))
    bits = 1 if kind == 'event' else 2
    if (set(h) != {'schema', 'channels', 'family', 'kind', 'sparse', 'bits', 'metadata'} or
            h['schema'] != 1 or h['channels'] != 32 or h['family'] != family or
            h['kind'] != kind or h['bits'] != bits or type(h['sparse']) is not bool or
            coords.dtype != np.dtype('<u2') or coords.ndim != 2 or coords.shape[1] != 3 or
            scales.dtype != np.dtype('<f4')):
        raise ValueError('Content packet contract mismatch')
    size = len(coords) * 32
    if h['sparse']:
        selected = unpack(mask, size, 1).astype(bool)
        decoded = unpack(values, int(selected.sum()), bits)
        if np.any(decoded == 0): raise ValueError('Redundant masked zero')
        codes = np.zeros(size, dtype=np.uint8); codes[selected] = decoded
    else:
        if mask.size: raise ValueError('Unexpected mask')
        codes = unpack(values, size, bits)
    codes = codes.reshape(-1, 32)
    _validate(coords, codes, scales, h['metadata'], family, kind)
    return coords, codes, scales, h['metadata']
