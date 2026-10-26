"""Common timing boundary, independent of repository-specific model loaders.

Adapters must execute the real encoder, selection, codec, receiver and NMS.
No packet file IO or reference validation occurs within the timed boundary.
This module is infrastructure, not an executed model performance result.
"""
import hashlib
import time
import platform
from pathlib import Path
import numpy as np


def selected_indices(size=1738, count=64):
    if size < count or count < 1:
        raise ValueError('Invalid common sample selection')
    return np.linspace(0, size-1, count, dtype=np.int64).tolist()


def measurement_indices(count):
    """Registered full measurement or its original four-point gate."""
    indices = selected_indices()
    if count == 64:
        return indices
    if count == 4:
        return [indices[i] for i in (0, 21, 42, 63)]
    raise ValueError('Only registered 4/64-frame selections are supported')


def host_metadata():
    cpuinfo = Path('/proc/cpuinfo')
    models = sorted(set(line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                        if line.startswith('model name'))) if cpuinfo.exists() else [platform.processor()]
    return dict(platform=platform.platform(),cpu_models=models,python=platform.python_version())


def array_signature(value):
    array = np.asarray(value)
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise ValueError('Predictions must be finite numeric arrays')
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def fingerprint(frame_id, arrays, packets):
    """No files or CUDA calls; execute only after the ending timestamp.

    Ordered packet labels identify direction/sender, not just byte length.
    Empty packets are valid for an explicitly supported zero-message mode.
    """
    if set(arrays) != {'boxes', 'scores'}:
        raise ValueError('Need CPU boxes and scores')
    boxes, scores = np.asarray(arrays['boxes']), np.asarray(arrays['scores'])
    if boxes.shape != (len(scores), 8, 3) or scores.shape != (len(scores),):
        raise ValueError('Unexpected prediction shape')
    seen = set(); rows = []
    for label, payload in packets:
        if not isinstance(label, str) or label in seen or not isinstance(payload, bytes):
            raise ValueError('Packet labels must be unique; payloads must be actual bytes')
        seen.add(label)
        rows.append(dict(label=label, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    return dict(frame_id=str(frame_id), boxes=array_signature(boxes), scores=array_signature(scores),
                packets=rows, total_bytes=sum(x['bytes'] for x in rows))


def require_reference(actual, reference):
    if actual != reference:
        raise ValueError('Prediction, identity, actual wire content or bytes changed')


def measure_once(adapter, cpu_batch, frame_id, reference, *, scope, synchronize,
                 clock=time.perf_counter):
    """Adapter methods: reset, to_device, infer, to_cpu, packets.

    to_cpu must return CPU NumPy boxes/scores (not deferred CUDA views).
    packets returns the actual in-memory payloads of this invocation, never
    precomputed reference messages. reset clears messages and per-call state.
    Caller supplies CUDA synchronization to allow CPU-only boundary tests.
    """
    if scope not in ('ready_cpu_batch_to_cpu_boxes', 'device_batch_to_cpu_boxes'):
        raise ValueError('Unknown timing scope')
    adapter.reset(frame_id)
    device_batch = adapter.to_device(cpu_batch) if scope == 'device_batch_to_cpu_boxes' else None
    synchronize()
    start = clock()
    if scope == 'ready_cpu_batch_to_cpu_boxes':
        device_batch = adapter.to_device(cpu_batch)
    result = adapter.infer(device_batch)
    arrays = adapter.to_cpu(result)
    synchronize()
    elapsed = clock() - start
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid elapsed time')
    # Actual payload extraction, hashing and validation are outside timing.
    actual = fingerprint(frame_id, arrays, adapter.packets())
    require_reference(actual, reference)
    return dict(frame_id=str(frame_id), seconds=elapsed, scope=scope,
                total_bytes=actual['total_bytes'], reference_exact=True)


def run_pass(adapter, indices, references, *, scope, synchronize, resource_snapshot,
             warmup=20, repeat=0, clock=time.perf_counter):
    """One pass; caller alternates methods/repeats and persists all raw rows.

    No filtering of outliers or resource snapshots. A failed equality check
    raises immediately; callers must preserve the failure and not auto-retry.
    GPU memory reset/measurement belongs to the concrete adapter/runner.
    """
    if not indices or len(set(indices)) != len(indices) or warmup < 0:
        raise ValueError('Need unique ordered frame indices and nonnegative warmup')
    before = resource_snapshot()
    rows = []
    for j in range(warmup + len(indices)):
        index = indices[j % len(indices)] if j < warmup else indices[j-warmup]
        frame_id, cpu_batch = adapter.prepare_cpu(index)
        reference = references[str(frame_id)]
        row = measure_once(adapter, cpu_batch, frame_id, reference, scope=scope,
                           synchronize=synchronize, clock=clock)
        if j >= warmup:
            row.update(index=index, repeat=repeat)
            rows.append(row)
    after = resource_snapshot()
    values = np.asarray([r['seconds'] for r in rows], dtype=np.float64)
    return dict(scope=scope, repeat=repeat, warmup_calls=warmup, frames=len(rows), rows=rows,
                seconds=dict(mean=float(values.mean()), median=float(np.median(values)),
                             p95=float(np.percentile(values, 95))),
                resource_before=before, resource_after=after,
                interference_filter_applied=False)
