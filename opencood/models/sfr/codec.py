"""Actual int8 wire codec, including masks, indices, scales and JSON metadata."""
import json
import struct
import numpy as np
import torch

MAGIC = b'SFR1'


def quantize(values):
    """Symmetric per-channel int8; final dimension is the channel dimension."""
    flat = values.reshape(-1, values.shape[-1])
    scale = flat.detach().abs().amax(0).clamp_min(1e-6)/127 if len(flat) else values.new_ones(values.shape[-1])/127
    integer = (values/scale).round().clamp(-127, 127).to(torch.int8)
    return integer, scale


def fake_quantize(values):
    integer, scale = quantize(values)
    decoded = integer.to(values.dtype)*scale
    return values + (decoded-values).detach()


def encode(metadata, tensors):
    arrays, entries, offset = [], {}, 0
    for name in sorted(tensors):
        array = tensors[name].detach().cpu().contiguous().numpy()
        raw = array.tobytes()
        entries[name] = {'dtype': array.dtype.str, 'shape': list(array.shape), 'offset': offset, 'bytes': len(raw)}
        arrays.append(raw)
        offset += len(raw)
    header = json.dumps({'metadata': metadata, 'tensors': entries}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    packet = MAGIC+struct.pack('<I', len(header))+header+b''.join(arrays)
    accounting = {k: v['bytes'] for k, v in entries.items()}
    accounting.update(header=8+len(header), total=len(packet))
    return packet, accounting


def decode(packet, device='cpu'):
    if packet[:4] != MAGIC or len(packet) < 8:
        raise ValueError('Not an SFR v1 message')
    length = struct.unpack('<I', packet[4:8])[0]
    header = json.loads(packet[8:8+length])
    payload = memoryview(packet)[8+length:]
    tensors = {}
    for name, entry in header['tensors'].items():
        start, size = entry['offset'], entry['bytes']
        dtype = np.dtype(entry['dtype'])
        if dtype.hasobject or start < 0 or size < 0 or start+size > len(payload) or np.prod(entry['shape'])*dtype.itemsize != size:
            raise ValueError('Invalid SFR tensor payload')
        array = np.frombuffer(payload[start:start+size], dtype=dtype).reshape(entry['shape']).copy()
        tensors[name] = torch.from_numpy(array).to(device)
    return header['metadata'], tensors
