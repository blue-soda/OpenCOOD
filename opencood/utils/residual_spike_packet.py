"""Ordered signed events; both SNN and ANN pay for identical wire fields.

One frame envelope, zlib ordered (step:uint8,address:uint32,value:int8).
This reference codec preserves logical steps, unlike a final count histogram.
No assertion of optimal compression or hardware execution efficiency.
"""
import json
import struct
import zlib
import numpy as np

HEADER = struct.Struct('<5sIII')
DTYPE = np.dtype([('step', 'u1'), ('address', '<u4'), ('value', 'i1')])


def encode(events, metadata):
    a = np.asarray(events, dtype=np.int64).reshape(-1, 3)
    if not len(a): return b''
    maximum = int(metadata['channels']) * int(np.prod(metadata['shape_zyx']))
    if (np.any(a[:, 0] < 0) or np.any(a[:, 0] >= metadata['steps']) or
            np.any(np.diff(a[:, 0]) < 0) or np.any(a[:, 1] < 0) or
            np.any(a[:, 1] >= maximum) or np.any(a[:, 2] == 0) or np.any(abs(a[:, 2]) > 3)):
        raise ValueError('Invalid step/address/signed value')
    # At most one symbol per address per logical step. Across steps repeats legal.
    if len(set(map(tuple, a[:, :2]))) != len(a): raise ValueError('Duplicate within step')
    raw = np.empty(len(a), dtype=DTYPE)
    for i, key in enumerate(DTYPE.names): raw[key] = a[:, i]
    meta = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    body = meta + zlib.compress(raw.tobytes())
    return HEADER.pack(b'E3RF1', len(meta), len(a), zlib.crc32(body)) + body


def decode(packet, metadata):
    if not packet:return np.empty((0,3), dtype=np.int64)
    if len(packet) < HEADER.size: raise ValueError('Truncated packet')
    magic, size, count, crc = HEADER.unpack_from(packet)
    body = packet[HEADER.size:]
    if magic != b'E3RF1' or zlib.crc32(body) != crc: raise ValueError('Bad version/CRC')
    expected = json.loads(json.dumps(metadata))
    if json.loads(body[:size].decode('utf8')) != expected: raise ValueError('Identity/geometry/basis mismatch')
    raw = zlib.decompress(body[size:])
    if len(raw) != count * DTYPE.itemsize: raise ValueError('Invalid event count')
    x = np.frombuffer(raw, dtype=DTYPE)
    a = np.column_stack([x[key].astype(np.int64) for key in DTYPE.names])
    if encode(a, metadata) != packet: raise ValueError('Noncanonical packet')
    return a


def append_prefix(previous, candidates, metadata, budget):
    """First-overflow stop, not a globally maximum zlib prefix claim.

    Every appended event is tested, so nonmonotonic compressed lengths cannot
    silently skip an overflowing prefix. Stops the frame, not just the step.
    """
    events = list(previous)
    packet = encode(events, metadata)
    if budget is not None and len(packet) > budget: raise ValueError('Existing prefix over budget')
    accepted = 0
    for row in candidates:
        trial = encode(events + [tuple(int(v) for v in row)], metadata)
        if budget is not None and len(trial) > budget: return events, packet, accepted, True
        events.append(tuple(int(v) for v in row)); packet = trial; accepted += 1
    return events, packet, accepted, False
