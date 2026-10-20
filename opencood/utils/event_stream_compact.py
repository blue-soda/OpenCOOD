"""Session-bound ordered event transport, with optional lossless zlib payload.

First packet carries all geometry/basis metadata. Subsequent packets carry its
SHA256, not an uncosted out-of-band session. No sorting or event coalescing.
This is a transport control; it does not change the IF representation.
"""
import hashlib
import json
import struct
import zlib
import numpy as np
from opencood.utils import event_stream_packet as reference

HEADER = struct.Struct('<5sBIIIII32s')
MAGIC = b'E3EC1'


def metadata_bytes(metadata):
    reference.validate_metadata(metadata)
    return json.dumps(metadata, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf8')


def encode(addresses, metadata, start, compress=True):
    context = metadata_bytes(metadata)
    if type(start) is not int or not 0 <= start < 2**32 or type(compress) is not bool:
        raise ValueError('Invalid compact offset or compression flag')
    ids = np.asarray(addresses)
    shape = metadata['contract']['shape_zyx']
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) == 0 or
            len(ids) > 1000000 or np.any(ids < 0) or np.any(ids >= 32 * np.prod(shape))):
        raise ValueError('Invalid event addresses')
    return _encode_validated(ids, context, start, compress)


def _encode_validated(ids, context, start, compress):
    raw = ids.astype('<u4').tobytes()
    payload = zlib.compress(raw) if compress else raw
    first = start == 0
    meta = context if first else b''
    body = meta + payload
    return HEADER.pack(MAGIC, int(first) | (int(compress) << 1), start, len(ids),
                       len(meta), len(payload), zlib.crc32(body),
                       hashlib.sha256(context).digest()) + body


def decode(packet, expected_metadata):
    context = metadata_bytes(expected_metadata)
    if len(packet) < HEADER.size:
        raise ValueError('Truncated compact header')
    magic, flags, start, count, msize, psize, crc, digest = HEADER.unpack(packet[:HEADER.size])
    if (magic != MAGIC or flags not in (0,1,2,3) or not 1 <= count <= 1000000 or
            bool(flags & 1) != (start == 0) or msize > 65536 or
            (start != 0 and msize != 0) or len(packet) != HEADER.size+msize+psize or
            digest != hashlib.sha256(context).digest()):
        raise ValueError('Wrong compact session/header')
    body = packet[HEADER.size:]
    if zlib.crc32(body) != crc or (start == 0 and body[:msize] != context):
        raise ValueError('Corrupt payload or changed initial metadata')
    payload = body[msize:]
    if flags & 2:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(payload, count*4+1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Invalid compressed event payload')
    else:
        raw = payload
    if len(raw) != count*4:
        raise ValueError('Wrong decoded event length')
    ids = np.frombuffer(raw, dtype='<u4').copy()
    if np.any(ids >= 32*np.prod(expected_metadata['contract']['shape_zyx'])):
        raise ValueError('Address outside registered grid')
    return ids, start


def pack_all(addresses, metadata, packet_events=256, compress=True):
    if type(packet_events) is not int or packet_events < 1:
        raise ValueError('Invalid packet capacity')
    metadata_bytes(metadata)
    return [encode(addresses[i:i+packet_events], metadata, i, compress)
            for i in range(0,len(addresses),packet_events)]


def pack_prefix(addresses, metadata, budget_bytes, packet_events=256, compress=True):
    """Exact greedy prefix admission; no monotonicity assumption about zlib.

Try each full block, then descend over every possible last-block length until
one fits. Slow reference, not a promised optimal number of packets.
"""
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError('Invalid budget')
    if type(packet_events) is not int or packet_events < 1:
        raise ValueError('Invalid packet capacity')
    context = metadata_bytes(metadata)
    addresses = np.asarray(addresses)
    if (addresses.ndim != 1 or not np.issubdtype(addresses.dtype,np.integer) or
            np.any(addresses < 0) or np.any(addresses >= 32*np.prod(metadata['contract']['shape_zyx'])) or
            len(addresses) >= 2**32 or packet_events > 1000000 or type(compress) is not bool):
        raise ValueError('Invalid event stream')
    packets, offset, used = [], 0, 0
    while offset < len(addresses):
        limit = min(packet_events, len(addresses)-offset)
        chosen = None
        for n in range(limit,0,-1):
            candidate = _encode_validated(addresses[offset:offset+n],context,offset,compress)
            if used+len(candidate) <= budget_bytes:
                chosen = candidate
                break
        if chosen is None:
            break
        packets.append(chosen); used += len(chosen); offset += n
        if n < limit:
            break
    return packets, offset, used


class CompactAccumulator(reference.StreamAccumulator):
    def receive(self, packet):
        ids, start = decode(packet,self.metadata)
        # Reuse exact identity, sequence, ceiling and atomic update validation.
        super().receive(reference.encode(ids,self.metadata,start))
