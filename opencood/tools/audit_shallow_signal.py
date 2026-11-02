"""Read-only signal-path diagnosis on fixed training frames, without AP or wire.

Counts are compared inside the same forward. Height-max survival uses changed
channel/XY columns as its denominator, not the number of changed 3D entries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.audit_snapshot_event_quota import no_codec
from opencood.tools.train_e3dsnn_single import write_json


def run(a):
    seed(); torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.3)
    output = Path(a.output)
    if output.exists():
        raise ValueError('Diagnostic output must be new')
    cfg = load_yaml(a.config)
    if cfg['model']['core_method'] != 'e3dsnn_snapshot_shallow_message':
        raise ValueError('This diagnosis measures the actual shallow height-max path')
    # Training IDs with inference preprocessing, no random augmentation or AP.
    data = build_dataset(cfg, train=False)
    training_split = data.manifest['splits']['train']
    if hashlib.sha256(Path(cfg['root_dir']).read_bytes()).hexdigest() != training_split['source_sha256']:
        raise ValueError('Training IDs must match the existing paired data audit')
    # Keep the validated inference preprocessor and select the independently
    # audited training IDs; do not mislabel a training file as the val split.
    data.data = list(training_split['valid_ids'])
    indices = np.linspace(0, len(data)-1, a.frames, dtype=np.int64).tolist()
    if a.frames < 2 or len(set(indices)) != a.frames:
        raise ValueError('Expected distinct evenly spaced training frames')
    model = train_utils.create_model(cfg).cuda().eval()
    model.snapshot_measure_bytes = model.packet_roundtrip = False
    model.snapshot_packet_directory = None
    observed = {}

    def receive(local, fused):
        delta = fused - local
        assert torch.isfinite(delta).all() and torch.equal(fused, fused.round())
        assert fused.min() >= 0 and fused.max() <= 4
        before, after = local.amax(2), fused.amax(2)
        columns = (delta != 0).any(2)
        changed = before != after
        assert not torch.any(changed & ~columns)
        observed.update(local=before, fused=after, counts=dict(
            entries_3d=local.numel(), changed_3d=int((delta != 0).sum()),
            columns=before.numel(), changed_columns_before_height=int(columns.sum()),
            changed_columns_after_height=int(changed.sum()),
            xy_pixels=changed.shape[-1]*changed.shape[-2],
            changed_xy_pixels=int(changed.any(1).sum())))

    def gate(module, inputs, output):
        gain = 2 * torch.sigmoid(output) * model.joint_receiver.current_scale
        observed['gain'] = gain

    def decode(module, inputs, output):
        aligned = inputs[0]
        current = output * observed['gain']
        observed['signal'] = dict(aligned_active_voxels=int((aligned != 0).any(1).sum()),
            grid_voxels=aligned.shape[2]*aligned.shape[3]*aligned.shape[4],
            current_abs_mean=float(current.abs().mean()), current_abs_max=float(current.abs().max()),
            gain_mean=float(observed['gain'].mean()))

    model.joint_receiver.snapshot_readout_observer = receive
    handles = [model.joint_receiver.local_gate.register_forward_hook(gate),
               model.content.expand.register_forward_hook(decode)]
    reports = []
    try:
        for label, path in (('initial', a.initial), ('adapted', a.checkpoint)):
            raw = Path(path).read_bytes()
            state = torch.load(path, map_location='cpu')
            model.load_state_dict(state, strict=True); model.eval()
            rows = []
            for i in indices:
                observed.clear()
                batch = train_utils.to_device(data.collate_batch_test([data[i]]), torch.device('cuda'))
                with no_codec(), torch.no_grad():
                    result = model(batch['ego'])
                    assert not result['communication_measured'] and 'packet_bytes' not in result
                    assert all(torch.isfinite(result[k]).all() for k in ('psm', 'rm', 'dm'))
                    detail = model.local_detail(observed['fused']) - model.local_detail(observed['local'])
                rows.append(dict(index=i, frame_id=str(batch['ego']['frame_ids'][0]),
                    events=int(result['stream_events'][0]), **observed['counts'], **observed['signal'],
                    detail_delta_abs_mean=float(detail.abs().mean()), detail_delta_abs_max=float(detail.abs().max())))
            totals = {k: sum(x[k] for x in rows) for k in observed['counts']}
            denom = totals['changed_columns_before_height']
            totals.update(height_survival_fraction=totals['changed_columns_after_height']/denom if denom else None,
                changed_3d_fraction=totals['changed_3d']/totals['entries_3d'],
                changed_2d_fraction=totals['changed_columns_after_height']/totals['columns'],
                changed_xy_fraction=totals['changed_xy_pixels']/totals['xy_pixels'])
            assert all(torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items())
            reports.append(dict(label=label, checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                current_scale=float(model.joint_receiver.current_scale), frames=rows, aggregate=totals,
                state_unchanged=True))
            write_json(output.with_name('progress.json'), dict(completed=label, samples=len(rows)))
    finally:
        for h in handles: h.remove()
        del model.joint_receiver.snapshot_readout_observer
    assert [x['frame_id'] for x in reports[0]['frames']] == [x['frame_id'] for x in reports[1]['frames']]
    write_json(output, dict(passed=True, no_codec=True, no_AP=True, no_optimizer=True,
        sampling='Evenly spaced training IDs, inference preprocessing, no augmentation',
        split_sha256=hashlib.sha256(Path(cfg['root_dir']).read_bytes()).hexdigest(),
        indexed_samples=len(data), arms=reports,
        limitation='Signal-path description only; no accuracy, causal bottleneck, or terminal architecture verdict'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'initial', 'checkpoint', 'output'): p.add_argument('--'+k, required=True)
    p.add_argument('--frames', type=int, default=32)
    run(p.parse_args())
