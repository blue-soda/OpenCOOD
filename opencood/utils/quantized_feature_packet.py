"""Eval-only rate probe: identical NPZ transport for Count4 and ANN PTQ.

ANN scales are sender-local per-frame/channel maxima, carried in every packet.
No labels, calibration set, receiver feature, or validation AP enters coding.
"""
import io
import json
import math
import numpy as np


def validate_options(options):
    if set(options) != {'bits', 'keep_fraction'}:
        raise ValueError('Expected bits and keep_fraction')
    if options['bits'] not in (2, 3, 4, 8):
        raise ValueError('Unsupported message bit width')
    fraction = float(options['keep_fraction'])
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Keep fraction must be in (0,1]')


def pack(values, bits):
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if np.any(values >= 2 ** bits):
        raise ValueError('Code exceeds bit width')
    binary = (values[:, None] >> np.arange(bits, dtype=np.uint8)) & 1
    return np.packbits(binary.reshape(-1), bitorder='little')


def unpack(payload, count, bits):
    if payload.dtype != np.uint8 or payload.size != (count * bits + 7) // 8:
        raise ValueError('Invalid packed length')
    binary = np.unpackbits(payload, bitorder='little')
    if np.any(binary[count * bits:]):
        raise ValueError('Nonzero padding')
    return (binary[:count * bits].reshape(-1, bits) *
            (1 << np.arange(bits, dtype=np.uint8))).sum(1).astype(np.uint8)


def prepare(coords, features, activation, options):
    """Return deterministic retained coordinates, codes, and transmitted scales."""
    validate_options(options)
    coords, features = np.asarray(coords), np.asarray(features, dtype=np.float32)
    if coords.shape != (len(features), 3) or features.shape != (len(coords), 128):
        raise ValueError('Expected [N,3] coordinates and [N,128] features')
    if (not np.isfinite(coords).all() or np.any(coords != np.floor(coords)) or
            np.any(coords < 0) or np.any(coords >= 65536) or
            len(np.unique(coords, axis=0)) != len(coords)):
        raise ValueError('Invalid coordinates')
    if not np.isfinite(features).all() or np.any(features < 0):
        raise ValueError('Expected finite nonnegative sender features')
    keep = np.any(features != 0, axis=1)
    coords, features = coords[keep], features[keep]
    # Coordinate tie-break makes selection independent of sparse CUDA row order.
    if len(coords):
        score = features.sum(1, dtype=np.float64)
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], -score))
        order = order[:int(math.ceil(len(order) * options['keep_fraction']))]
        coords, features = coords[order], features[order]
    bits = options['bits']
    if activation == 'count4':
        if bits != 3 or np.any(features != np.floor(features)) or np.any(features > 4):
            raise ValueError('Lossless Count4 requires three-bit integer codes')
        codes, scales = features.astype(np.uint8), np.ones(128, dtype='<f4')
    elif activation == 'relu':
        maxima = features.max(0) if len(features) else np.zeros(128, dtype=np.float32)
        scales = np.maximum(maxima / (2 ** bits - 1), np.finfo(np.float32).tiny).astype('<f4')
        codes = np.floor(np.clip(features / scales, 0, 2 ** bits - 1) + .5).astype(np.uint8)
    else:
        raise ValueError('Unknown sender activation')
    keep = np.any(codes != 0, axis=1)
    coords, codes = coords[keep].astype('<u2'), codes[keep]
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    return coords[order], codes[order], scales


def encode(coords, features, metadata, activation, options):
    coords, codes, scales = prepare(coords, features, activation, options)
    bits = options['bits']
    # Same two candidate transports for both model families, actual ZIP bytes.
    candidates = []
    mask = codes != 0
    for sparse in (False, True):
        stream = io.BytesIO()
        np.savez_compressed(stream, coords=coords, scales=scales,
            codes=pack(codes[mask] if sparse else codes, bits),
            mask=np.packbits(mask.reshape(-1), bitorder='little') if sparse else np.empty(0, dtype=np.uint8),
            header=np.asarray(json.dumps(dict(schema=1, bits=bits, sparse=sparse,
                activation=activation, options=options, metadata=metadata),
                sort_keys=True, separators=(',', ':'), allow_nan=False)))
        candidates.append(stream.getvalue())
    return min(candidates, key=len)


def decode(packet, activation, options):
    validate_options(options)
    with np.load(io.BytesIO(packet), allow_pickle=False) as data:
        coords, scales = data['coords'].copy(), data['scales'].copy()
        payload, mask_payload = data['codes'].copy(), data['mask'].copy()
        header = json.loads(str(data['header'].item()))
    if (header['schema'] != 1 or header['activation'] != activation or
            header['options'] != options or header['bits'] != options['bits']):
        raise ValueError('Wire contract mismatch')
    shape = np.asarray(header['metadata']['shape_zyx'])
    if (shape.shape != (3,) or np.any(shape <= 0) or np.prod(shape) > 1000000 or
            coords.ndim != 2 or coords.shape[1] != 3 or coords.dtype != np.dtype('<u2') or
            np.any(coords >= shape) or len(np.unique(coords, axis=0)) != len(coords) or
            scales.shape != (128,) or scales.dtype != np.dtype('<f4') or
            not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError('Invalid decoded grid or scales')
    size, bits = len(coords) * 128, options['bits']
    if header['sparse']:
        mask = unpack(mask_payload, size, 1).astype(bool)
        values = unpack(payload, int(mask.sum()), bits)
        if np.any(values == 0): raise ValueError('Redundant masked zero')
        codes = np.zeros(size, dtype=np.uint8); codes[mask] = values
    else:
        if mask_payload.size: raise ValueError('Unexpected mask')
        codes = unpack(payload, size, bits)
    codes = codes.reshape(-1, 128)
    if np.any(~np.any(codes != 0, axis=1)):
        raise ValueError('Redundant zero row')
    if activation == 'count4' and (bits != 3 or np.any(codes > 4) or not np.all(scales == 1)):
        raise ValueError('Invalid Count4 contract')
    return coords, codes.astype(np.float32) * scales, header['metadata']
