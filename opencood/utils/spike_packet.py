"""Lossless Count4 feature packets for one sender and one query.

Coordinates are uint16 z/y/x indices, not calibrated world positions. Frames
carry JSON metadata and CRC32. Transport fragmentation/security is out of scope.
"""
import json
import struct
import zlib
import numpy as np

HEADER = struct.Struct('<4sBBHIII')


def _pack3(values):
    bits = (values.reshape(-1, 1).astype(np.uint8) >> np.arange(3, dtype=np.uint8)) & 1
    return np.packbits(bits.ravel(), bitorder='little').tobytes()


def _unpack3(payload, count):
    if len(payload) != (count * 3 + 7) // 8:
        raise ValueError('Incorrect count payload length')
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder='little')
    if np.any(bits[count * 3:]):
        raise ValueError('Nonzero padding bits')
    values = (bits[:count * 3].reshape(-1, 3) * np.array([1, 2, 4], dtype=np.uint8)).sum(1).astype(np.uint8)
    if np.any(values > 4):
        raise ValueError('Count4 value outside 0..4')
    return values


def encode(coords, features, metadata, mode='auto'):
    """Prune all-zero rows and pack dense 3-bit counts or a mask + nonzeros."""
    coords, features = np.asarray(coords), np.asarray(features)
    if coords.ndim != 2 or coords.shape[1] != 3 or features.ndim != 2 or len(coords) != len(features):
        raise ValueError('Expected coordinates [M,3] and features [M,C]')
    if not 0 < features.shape[1] <= 4096:
        raise ValueError('Invalid channel count')
    if not np.isfinite(coords).all() or np.any(coords != np.floor(coords)) or np.any(coords < 0) or np.any(coords >= 65536):
        raise ValueError('Coordinates must fit uint16')
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError('Duplicate spatial address')
    if not np.isfinite(features).all() or np.any(features != np.floor(features)) or np.any(features < 0) or np.any(features > 4):
        raise ValueError('Expected integer-valued Count4 features')
    if mode not in ('dense3', 'mask3', 'auto'):
        raise ValueError('Unknown codec mode')
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    keep = np.any(features != 0, axis=1)
    coords = coords[keep].astype('<u2')
    values = features[keep].astype(np.uint8)
    # Both packet lengths are known before packing. Avoid constructing the
    # much larger dense candidate when channel sparsity favors mask3.
    kind = 0 if mode == 'dense3' else 1
    if mode == 'auto':
        dense_size = (values.size * 3 + 7) // 8
        mask_size = (values.size + 7) // 8 + (np.count_nonzero(values) * 3 + 7) // 8
        kind = 0 if dense_size <= mask_size else 1
    if kind == 0:
        payload = coords.tobytes() + _pack3(values)
    else:
        mask = values != 0
        payload = coords.tobytes() + np.packbits(mask.ravel(), bitorder='little').tobytes() + _pack3(values[mask])
    body = HEADER.pack(b'SPK4', 1, kind, features.shape[1], len(coords), len(metadata_bytes), len(payload)) + metadata_bytes + payload
    return body + struct.pack('<I', zlib.crc32(body) & 0xffffffff)


def decode(packet):
    """Return sender-local coordinates, uint8 counts and explicit metadata."""
    if len(packet) < HEADER.size + 4:
        raise ValueError('Truncated packet')
    body, checksum = packet[:-4], packet[-4:]
    if zlib.crc32(body) & 0xffffffff != struct.unpack('<I', checksum)[0]:
        raise ValueError('Packet checksum mismatch')
    magic, version, kind, channels, rows, metadata_size, payload_size = HEADER.unpack_from(body)
    if magic != b'SPK4' or version != 1 or kind not in (0, 1) or not 0 < channels <= 4096:
        raise ValueError('Unsupported packet format')
    if len(body) != HEADER.size + metadata_size + payload_size or rows * channels > 100000000:
        raise ValueError('Invalid packet size')
    metadata = json.loads(body[HEADER.size:HEADER.size + metadata_size].decode('utf-8'))
    payload = body[HEADER.size + metadata_size:]
    if len(payload) < rows * 6:
        raise ValueError('Truncated coordinates')
    coords = np.frombuffer(payload[:rows * 6], dtype='<u2').reshape(rows, 3).copy()
    if len(np.unique(coords, axis=0)) != rows:
        raise ValueError('Duplicate spatial address')
    rest = payload[rows * 6:]
    if kind == 0:
        values = _unpack3(rest, rows * channels).reshape(rows, channels)
    else:
        mask_size = (rows * channels + 7) // 8
        if len(rest) < mask_size:
            raise ValueError('Truncated channel mask')
        bits = np.unpackbits(np.frombuffer(rest[:mask_size], dtype=np.uint8), bitorder='little')
        if np.any(bits[rows * channels:]):
            raise ValueError('Nonzero mask padding')
        mask = bits[:rows * channels].reshape(rows, channels).astype(bool)
        values = np.zeros((rows, channels), dtype=np.uint8)
        values[mask] = _unpack3(rest[mask_size:], int(mask.sum()))
        if np.any(values[mask] == 0):
            raise ValueError('Zero value marked as nonzero')
    if np.any(~np.any(values != 0, axis=1)):
        raise ValueError('Redundant zero row')
    return coords, values, metadata
