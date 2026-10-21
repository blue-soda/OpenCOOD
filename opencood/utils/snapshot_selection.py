"""Conservative first-overflow selection under actual E3GS1 snapshot bytes.

Both treatments share the same IF stream and codec. Complete spatial rows use
their first event to rank them, but include the full frame's counts: offline
frame scheduling, NOT a causal online spike policy. No monotonic-size assumption.
"""
import json
import time
import zlib
import numpy as np
from opencood.tools.profile_e3dsnn_event_granularity import HEADER, decode_snapshot
from opencood.utils.event_stream_packet import validate_metadata
from opencood.utils.quantized_feature_packet import pack


def encode_rows(coords, rows, metadata):
    """Byte-identical sparse construction of the existing dense snapshot codec."""
    contract = validate_metadata(metadata)
    coords, rows = np.asarray(coords), np.asarray(rows)
    if (coords.ndim != 1 or rows.shape != (len(coords), 32) or
            not np.isfinite(coords).all() or np.any(coords != np.floor(coords)) or
            np.any(coords < 0) or np.any(coords >= np.prod(contract.shape_zyx)) or
            np.any(coords[1:] <= coords[:-1]) or metadata['max_spikes'] > 15 or
            not np.isfinite(rows).all() or np.any(rows != np.floor(rows)) or
            np.any(rows < 0) or np.any(rows > metadata['max_spikes']) or
            not np.all(np.any(rows != 0, axis=1))):
        raise ValueError('Invalid sorted snapshot rows')
    if not len(coords):
        return b''
    mask = rows != 0; values = rows[mask]
    raw = coords.astype('<u4').tobytes() + pack(mask, 1).tobytes() + pack(values, 4).tobytes()
    context = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    body = context + zlib.compress(raw)
    return HEADER.pack(b'E3GS1', len(context), len(coords), len(values), len(raw), zlib.crc32(body)) + body


def select_snapshot(addresses, metadata, budget, mode):
    """Check each prefix, stop at its FIRST overflow; not maximal feasible search."""
    started = time.perf_counter()
    contract = validate_metadata(metadata)
    size = int(np.prod(contract.shape_zyx))
    addresses = np.asarray(addresses)
    if (addresses.ndim != 1 or not np.issubdtype(addresses.dtype, np.integer) or
            np.any(addresses < 0) or np.any(addresses >= 32 * size) or
            type(budget) is not int or budget < 0 or mode not in ('events', 'rows')):
        raise ValueError('Invalid stream, budget or selection mode')
    # Full content validation is independent of which prefix fits.
    full = np.bincount(addresses.astype(np.int64), minlength=32 * size).reshape(32, size).T
    if np.any(full > metadata['max_spikes']):
        raise ValueError('Event stream exceeds declared ceiling')
    spatial = addresses % size
    coords = np.unique(spatial)
    lookup = np.searchsorted(coords, spatial)
    if mode == 'events':
        order = range(len(addresses))
    else:
        _, first = np.unique(spatial, return_index=True)
        order = np.sort(first)
    selected = np.zeros((len(coords), 32), dtype=np.uint8)
    packet = b''; accepted = 0; rejected_bytes = None; attempts = 0
    for event_index in order:
        slot = lookup[event_index]
        previous = selected[slot].copy()
        if mode == 'events':
            selected[slot, int(addresses[event_index]) // size] += 1
        else:
            selected[slot] = full[coords[slot]]
        mask = np.any(selected != 0, axis=1)
        candidate = encode_rows(coords[mask], selected[mask], metadata)
        attempts += 1
        if len(candidate) > budget:
            selected[slot] = previous
            rejected_bytes = len(candidate)
            break
        packet = candidate
        accepted += 1
    mask = np.any(selected != 0, axis=1)
    assert packet == encode_rows(coords[mask], selected[mask], metadata)
    decoded = decode_snapshot(packet, metadata)
    restored = decoded.reshape(32, size).T
    assert np.array_equal(restored[coords], selected)
    assert int(np.count_nonzero(restored)) == int(np.count_nonzero(selected))
    assert len(packet) <= budget
    if mode == 'rows':
        assert np.array_equal(selected[mask], full[coords[mask]])
    return packet, decoded, dict(mode=mode, budget_bytes=budget, actual_bytes=len(packet),
        unused_bytes=budget-len(packet), accepted_units=accepted, attempted_prefixes=attempts,
        first_rejected_bytes=rejected_bytes, selected_events=int(selected.sum()),
        selected_voxels=int(mask.sum()), selected_channels=int(np.count_nonzero(selected)),
        selection_seconds=time.perf_counter()-started,
        semantics='first-overflow stop; not guaranteed globally maximal feasible prefix')
