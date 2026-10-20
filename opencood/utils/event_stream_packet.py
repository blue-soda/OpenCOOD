"""Auditable raw address-event wire reference with exact byte-budget admission.

Each unit event is one little-endian uint32 flattened CZYX address. Explicit
per-packet identity/geometry metadata and sequence offsets are counted.
No assumed compression gains; compressed transport is a later controlled
comparison. Order is preserved; repeated addresses are legal.
"""
import json
import struct
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract, check_transform

MAGIC = b'E3EV1'


def validate_metadata(metadata):
    if set(metadata) != {'contract', 'source_to_ego', 'channels', 'max_spikes', 'schema'}:
        raise ValueError('Incomplete event identity')
    if metadata['schema'] != 'if-unit-events-v1' or metadata['channels'] != 32:
        raise ValueError('Unknown event schema')
    if type(metadata['max_spikes']) is not int or not 1 <= metadata['max_spikes'] <= 64:
        raise ValueError('Invalid event ceiling')
    raw = dict(metadata['contract'])
    for key in ('shape_zyx', 'origin_xyz', 'step_xyz', 'timestamps_us'):
        raw[key] = tuple(raw[key])
    contract = MessageContract(**raw)
    basis = contract.channel_basis
    if (not basis.startswith('event-content32-') or len(basis) != 80 or
            any(c not in '0123456789abcdef' for c in basis[16:])):
        raise ValueError('Invalid stream basis digest')
    if 32 * int(np.prod(contract.shape_zyx)) >= 2**32:
        raise ValueError('Grid exceeds address space')
    check_transform(torch.tensor(metadata['source_to_ego'], dtype=torch.float64)[None])
    return contract


def encode(addresses, metadata, start):
    contract = validate_metadata(metadata)
    if type(start) is not int or start < 0:
        raise ValueError('Invalid sequence offset')
    ids = np.asarray(addresses)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or
            np.any(ids < 0) or np.any(ids >= 32 * np.prod(contract.shape_zyx))):
        raise ValueError('Invalid channel/spatial address')
    header = json.dumps(dict(metadata=metadata, start=start, count=len(ids)),
                        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    return MAGIC + struct.pack('<I', len(header)) + header + ids.astype('<u4').tobytes()


def decode(packet):
    if len(packet) < 9 or packet[:5] != MAGIC:
        raise ValueError('Wrong event packet')
    size = struct.unpack('<I', packet[5:9])[0]
    if size > 65536 or len(packet) < 9 + size:
        raise ValueError('Invalid header size')
    header = json.loads(packet[9:9+size].decode('utf8'))
    if set(header) != {'metadata', 'start', 'count'}:
        raise ValueError('Invalid header fields')
    if type(header['count']) is not int or header['count'] < 0:
        raise ValueError('Invalid event count')
    if len(packet) != 9 + size + 4 * header['count']:
        raise ValueError('Truncated or trailing payload')
    ids = np.frombuffer(packet[9+size:], dtype='<u4').copy()
    # Canonical form also validates all header fields and addresses.
    if encode(ids, header['metadata'], header['start']) != packet:
        raise ValueError('Noncanonical event packet')
    return ids, header['metadata'], header['start']


def pack_prefix(addresses, metadata, budget_bytes, packet_events=256):
    """Largest sequential prefix fitting fixed-record packets; never overspend.

packet_events is a transport limit, not a neural timestep. Empty streams send
zero packets. Different packet limits change overhead and budget admission,
so regrouping equivalence must hold at equal events, not equal total bytes.
"""
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError('Invalid byte budget')
    if type(packet_events) is not int or packet_events < 1:
        raise ValueError('Invalid packet capacity')
    validate_metadata(metadata)
    packets, offset, used = [], 0, 0
    while offset < len(addresses):
        lo, hi = 0, min(packet_events, len(addresses) - offset)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            packet = encode(addresses[offset:offset+mid], metadata, offset)
            if used + len(packet) <= budget_bytes:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            break
        packet = encode(addresses[offset:offset+lo], metadata, offset)
        packets.append(packet); used += len(packet); offset += lo
    return packets, offset, used


class StreamAccumulator:
    """Integer source-grid session; duplicate sequence ranges are rejected."""
    def __init__(self, metadata):
        contract = validate_metadata(metadata)
        self.metadata = json.loads(json.dumps(metadata))
        self.counts = np.zeros((1, 32) + contract.shape_zyx, dtype=np.int64)
        self.offset, self.closed = 0, False

    def receive(self, packet):
        ids, metadata, start = decode(packet)
        if self.closed or metadata != self.metadata or start != self.offset or len(ids) == 0:
            raise ValueError('Wrong frame/basis/pose, closed stream or nonsequential events')
        unique, frequency = np.unique(ids, return_counts=True)
        flat = self.counts.ravel()
        if np.any(flat[unique] + frequency > self.metadata['max_spikes']):
            raise ValueError('Per-address spike ceiling exceeded')
        flat[unique] += frequency
        self.offset += len(ids)

    def finish(self):
        if self.closed:
            raise ValueError('Closed stream')
        self.closed = True
        return self.counts.copy()
