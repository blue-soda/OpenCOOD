"""Count MACs, spikes and neuron updates for one frame, then estimate energy.

All energy numbers are estimates under stated hardware assumptions; the measured
quantities are the operation counts. Two assumption sets are reported (a
low/high e_bit pair and the two synaptic energies) so that no single point
estimate is presented as a measurement.
"""
import argparse
import json
from pathlib import Path

import torch

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike

E_MAC_FP32_PJ = 4.6
E_MAC_INT8_PJ = 0.2
E_SYN_PJ = (0.9, 1.74)
E_NEU_PJ = 23.6
E_BIT_NJ = (0.1, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--bytes', type=float, default=10815.4,
                        help='Measured mean bytes per frame for the communication term')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = yaml_utils.load_yaml(args.config)
    cfg['model']['args']['snapshot']['packet_directory'] = None
    device = torch.device('cuda:0')
    model = train_utils.create_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    model.to(device).eval()
    dataset = build_dataset(cfg, train=False)

    spikes, neurons = [], []

    def hook(module, inputs, output):
        spikes.append(float(output.detach().float().sum()))
        neurons.append(float(inputs[0].numel()))

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, Multispike)]
    frame_flops, frame_spikes, frame_neurons = [], [], []
    try:
        with torch.no_grad():
            for index in range(args.frames):
                batch = train_utils.to_device(dataset.collate_batch_test([dataset[index]]), device)
                if index == 0:
                    with torch.profiler.profile(with_flops=True) as prof:
                        model(batch['ego'])
                    flops = sum(e.flops for e in prof.key_averages() if getattr(e, 'flops', 0))
                    frame_flops.append(flops)
                else:
                    model(batch['ego'])
                frame_spikes.append(sum(spikes))
                frame_neurons.append(sum(neurons))
                spikes.clear()
                neurons.clear()
    finally:
        for handle in handles:
            handle.remove()

    macs = sum(frame_flops) / 2.0 if frame_flops else 0.0
    spike_mean = sum(frame_spikes) / len(frame_spikes)
    neuron_mean = sum(frame_neurons) / len(frame_neurons)
    comm_low = args.bytes * 8 * E_BIT_NJ[0] / 1000.0   # nJ -> pJ
    comm_high = args.bytes * 8 * E_BIT_NJ[1] / 1000.0
    compute_mac = macs * E_MAC_FP32_PJ
    compute_spike_low = spike_mean * E_SYN_PJ[0] + neuron_mean * E_NEU_PJ
    compute_spike_high = spike_mean * E_SYN_PJ[1] + neuron_mean * E_NEU_PJ
    result = {
        'frames': args.frames,
        'macs_per_frame': macs,
        'spikes_per_frame': spike_mean,
        'neuron_updates_per_frame': neuron_mean,
        'bytes_per_frame': args.bytes,
        'energy_pj': {
            'communication_low': comm_low,
            'communication_high': comm_high,
            'ann_compute_fp32': compute_mac,
            'snn_compute_low': compute_spike_low,
            'snn_compute_high': compute_spike_high,
            'total_low': compute_mac + compute_spike_low + comm_low,
            'total_high': compute_mac + compute_spike_high + comm_high,
        },
        'energy_uj': {k: v / 1e6 for k, v in {
            'communication_low': comm_low,
            'communication_high': comm_high,
            'ann_compute_fp32': compute_mac,
            'snn_compute_low': compute_spike_low,
            'snn_compute_high': compute_spike_high,
            'total_low': compute_mac + compute_spike_low + comm_low,
            'total_high': compute_mac + compute_spike_high + comm_high,
        }.items()},
        'assumptions': {
            'e_mac_fp32_pj': E_MAC_FP32_PJ,
            'e_syn_pj': list(E_SYN_PJ),
            'e_neu_pj': E_NEU_PJ,
            'e_bit_nj': list(E_BIT_NJ),
            'note': 'Operation counts are measured; energy values are estimates under these assumptions',
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
