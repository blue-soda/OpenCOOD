"""Explicit one-way count transport for the single-pass geometry backend.

Three full latent count grids and channel scales are paid. Geometric support
is derived at the receiver, with no confidence mask, request or late boxes.
This audit runs only during explicit evaluation, not training or timing studies.
"""
import hashlib
import struct
from pathlib import Path
import numpy as np
from cosdh_count_wire import CoSDHCountWire, encode_counts, decode_counts
from cosdh_wire import HEADER
ENVELOPE = struct.Struct('<QII16d')


def envelope(frame_id, sender, pose):
    transform = np.asarray(pose, dtype='<f8')
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Finite 4x4 pose required')
    return ENVELOPE.pack(int(frame_id), sender, 0, *transform.reshape(-1))


CONTAINER = struct.Struct('<4sI')


def decode_response(packet):
    if len(packet) < ENVELOPE.size + CONTAINER.size:
        raise ValueError('Truncated geometry count response')
    header = ENVELOPE.unpack_from(packet)
    if header[1:3] != (1, 0):
        raise ValueError('Expected roadside sender and sequence zero')
    magic, number = CONTAINER.unpack_from(packet, ENVELOPE.size)
    if magic != b'CGW1' or number != 3:
        raise ValueError('Expected three geometry count stages')
    offset = ENVELOPE.size + CONTAINER.size
    stages = []
    for expected in range(3):
        if len(packet) < offset + HEADER.size:
            raise ValueError('Truncated count section header')
        fields = HEADER.unpack_from(packet, offset)
        end = offset + HEADER.size + fields[-2]
        if end > len(packet):
            raise ValueError('Truncated count section body')
        stage, steps, counts, scales = decode_counts(packet[offset:end])
        if stage != expected:
            raise ValueError('Wrong stage order')
        stages.append((steps, counts, scales))
        offset = end
    if offset != len(packet):
        raise ValueError('Trailing geometry response bytes')
    pose = np.asarray(header[3:], dtype='f8').reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError('Nonfinite pose')
    return header[0], pose, stages


class CoSDHGeometryCountWire(CoSDHCountWire):
    def __init__(self, model, counts, output):
        # Register after the count converter so the input is already quantized.
        self.counts = counts
        self.output = Path(output) / 'wire'
        self.output.mkdir()
        self.handles = [model.naive_compressor_list[i].decoder.register_forward_pre_hook(
            self.count_hook(i)) for i in range(3)]

    def begin(self, batch, frame_id):
        import torch
        super().begin(batch, frame_id)
        self.prefix = envelope(frame_id, 1, self.pose)
        restored = np.asarray(ENVELOPE.unpack(self.prefix)[3:], dtype='f8').reshape(4, 4)
        assert np.array_equal(restored, self.pose), 'Envelope changed receiver pose'
        # The actual downstream normalization/warp consumes the decoded pose.
        transform = batch['ego']['pairwise_t_matrix']
        decoded = torch.from_numpy(restored.copy()).to(device=transform.device, dtype=transform.dtype)
        assert torch.equal(decoded, transform[0, 0, 1])
        transform[0, 0, 1] = decoded

    def finish(self, index):
        assert sorted(self.parts) == [0, 1, 2] and self.request is None
        packet = self.prefix + CONTAINER.pack(b'CGW1', 3) + b''.join(self.parts[i] for i in range(3))
        frame, pose, stages = decode_response(packet)
        assert frame == ENVELOPE.unpack(self.prefix)[0] and np.array_equal(pose, self.pose)
        for stage, (steps, counts, scales) in enumerate(stages):
            assert steps == self.counts.steps
            assert np.array_equal(counts, self.counts.current[stage].cpu().numpy())
            assert np.array_equal(scales, self.counts.scales[stage].cpu().numpy().reshape(-1))
        path = self.output / ('%06d_response.bin' % index)
        path.write_bytes(packet)
        row = dict(direction='response', file='wire/' + path.name, bytes=len(packet),
                   sha256=hashlib.sha256(packet).hexdigest())
        return dict(packets=[row], demand_bytes=0, response_bytes=len(packet), total_bytes=len(packet),
                    envelope_bytes=ENVELOPE.size, container_bytes=CONTAINER.size,
                    late_boxes=0, late_section_bytes=0, mask_bytes=0,
                    receiver_pose_decoded=True, support_derived_from_geometry=True,
                    sections=[self.details[i] for i in range(3)])


def self_test():
    rng = np.random.RandomState(303)
    for steps in (8, 32):
        arrays = [np.zeros((2, 3, 5), dtype='u1'), rng.randint(0, steps+1, (3, 2, 7)),
                  np.full((1, 4, 3), steps, dtype='u1')]
        scales = [np.linspace(.1, 2., len(x)).astype('f4') for x in arrays]
        pose = np.eye(4, dtype='f8'); pose[0, 3] = 1.25 + 2.**-40
        parts = [encode_counts(i, x, scales[i], steps) for i, x in enumerate(arrays)]
        prefix = envelope('000123', 1, pose)
        packet = prefix + CONTAINER.pack(b'CGW1', 3) + b''.join(parts)
        frame, got_pose, got = decode_response(packet)
        assert frame == 123 and np.array_equal(pose, got_pose)
        for i, (t, x, s) in enumerate(got):
            assert t == steps and np.array_equal(x, arrays[i]) and np.array_equal(s, scales[i])
        assert len(packet) == 144 + 8 + sum(map(len, parts))
        for bad in (packet[:-1], packet+b'x', prefix+CONTAINER.pack(b'CGW1', 3)+b''.join(parts[::-1])):
            try:
                decode_response(bad)
            except (ValueError, AssertionError):
                pass
            else:
                raise AssertionError('Malformed container accepted')
    print('Geometry count container, pose, scales, stages and malformed packets passed')


if __name__ == '__main__':
    self_test()
