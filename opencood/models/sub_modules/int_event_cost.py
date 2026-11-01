"""Read-only hypothetical event costs for the frozen two-conv readout.

Counts synaptic deliveries, not executed CUDA operations. Supported geometry:
3x3, stride1, dilation1, groups1, same zero padding. No sparse kernel here.
"""
import torch


def spatial_event_cost(x, scales, steps, initial_membrane_units, out_channels):
    if x.ndim != 4 or scales.shape != (1, x.shape[1], 1, 1):
        raise ValueError('Expected NCHW and per-input-channel scales')
    if min(x.shape) < 1 or steps < 1 or out_channels < 1:
        raise ValueError('Nonempty geometry and positive steps/channels required')
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError('Finite activations and positive finite scales required')
    q = 65536
    if initial_membrane_units not in (0, q//2):
        raise ValueError('Only frozen zero/half initial membrane supported')
    positive = x.clamp_min(0)
    exceptions = positive > scales
    drive = torch.floor(torch.minimum(positive, scales)/scales*q).to(torch.int64).clamp_(0, q)
    drive.masked_fill_(exceptions, 0)
    counts = (drive*steps+initial_membrane_units)//q
    h, w = x.shape[-2:]
    # Number of valid output positions receiving each input, including H/W=1.
    ys = torch.arange(h, device=x.device)
    xs = torch.arange(w, device=x.device)
    ry = 1+(ys > 0).long()+(ys+1 < h).long()
    rx = 1+(xs > 0).long()+(xs+1 < w).long()
    fanout = ry[:, None]*rx[None, :]*out_channels
    spikes = int(counts.sum())
    tail = int(exceptions.sum())
    active = int((counts > 0).sum())
    return dict(shape=list(x.shape), out_channels=out_channels,
                spike_count=spikes, analog_bypass_elements=tail, active_count_elements=active,
                event_ac_exact=int((counts*fanout).sum()), event_ac_upper=spikes*9*out_channels,
                analog_mac_exact=int((exceptions.long()*fanout).sum()), analog_mac_upper=tail*9*out_channels,
                sparse_count_mac_exact=int(((counts > 0).long()*fanout).sum()),
                scope='Hypothetical valid synaptic deliveries; zero padding skipped, zero weights not skipped; not GPU executed ops')


def attach_spatial_event_audit(readout):
    """Attach only after source geometry validation; do not change outputs."""
    handles = []
    convs = readout.body.layers[0].double_conv
    for codec, conv in ((readout.input_codec, convs[0]), (readout.middle_codec, convs[3])):
        assert conv.kernel_size == (3, 3) and conv.stride == (1, 1)
        assert conv.padding == (1, 1) and conv.dilation == (1, 1) and conv.groups == 1
        assert conv.padding_mode == 'zeros' and codec.preserve_tail and codec.mode == 'count'

        def hook(module, inputs, output, channels=conv.out_channels):
            cost = spatial_event_cost(inputs[0], module.scales, module.steps, module.initial_membrane_units, channels)
            assert cost['spike_count'] == module.last_stats['spike_count']
            assert cost['analog_bypass_elements'] == module.last_stats['analog_bypass_elements']
            assert cost['event_ac_exact'] <= cost['event_ac_upper']
            assert cost['analog_mac_exact'] <= cost['analog_mac_upper']
            assert cost['sparse_count_mac_exact'] <= cost['event_ac_exact']
            module.last_stats['spatial_cost'] = cost

        handles.append(codec.register_forward_hook(hook))
    return handles
