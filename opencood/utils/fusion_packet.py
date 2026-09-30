"""Receiver reconstructs a 3D grid from a sender-local feature packet.

SNN uses the existing lossless Count4 codec; ANN uses lossless FP32 NPZ.
This is a transport correctness baseline, not an equal-rate quantization study.
"""
import io
import json
import numpy as np
from opencood.utils import spike_packet


def encode(coords, features, metadata, activation):
    if activation == 'count4':
        return spike_packet.encode(coords, features, metadata)
    if activation != 'relu':
        raise ValueError('Unknown feature activation')
    keep = np.any(features != 0, axis=1)
    stream = io.BytesIO()
    np.savez_compressed(stream, coords=np.asarray(coords[keep], dtype='<u2'),
                        features=np.asarray(features[keep], dtype='<f4'),
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    return stream.getvalue()


def decode(packet, activation):
    if activation == 'count4':
        coords, features, meta = spike_packet.decode(packet)
    elif activation == 'relu':
        with np.load(io.BytesIO(packet), allow_pickle=False) as data:
            coords, features = data['coords'].copy(), data['features'].copy()
            meta = json.loads(str(data['metadata'].item()))
    else:
        raise ValueError('Unknown feature activation')
    shape = np.asarray(meta['shape_zyx'])
    if coords.shape != (len(features), 3) or features.shape != (len(coords), 128):
        raise ValueError('Invalid feature packet dimensions')
    if (shape.shape != (3,) or np.any(shape <= 0) or np.prod(shape) > 1000000
            or np.any(coords < 0) or np.any(coords >= shape)
            or len(np.unique(coords, axis=0)) != len(coords)
            or not np.isfinite(features).all()):
        raise ValueError('Invalid coordinates/values in feature packet')
    return coords, features, meta
