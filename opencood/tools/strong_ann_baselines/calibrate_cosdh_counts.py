"""Training-only CoSDH sender-latent calibration before count conversion.

No AP, optimizer, packets or model mutation. The next 32 training frames audit
fixed thresholds fitted on the first 64; validation IDs are forbidden.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'model-dir', 'data-root', 'train-split', 'val-split', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--interface', choices=('sender', 'post_fusion'), default='sender')
    p.add_argument('--checkpoint-name', default='net_epoch11.pth')
    p.add_argument('--sender-thresholds')
    a = p.parse_args()
    if a.interface == 'post_fusion' and not a.sender_thresholds:
        p.error('Post-fusion calibration requires the fixed IF8 sender thresholds')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    repo = Path(a.repo).resolve(); sys.path.insert(0, str(repo)); os.chdir(str(repo))
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset

    random.seed(303); np.random.seed(303); torch.manual_seed(303)
    torch.cuda.manual_seed_all(303); torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.25)
    train_ids = json.loads(Path(a.train_split).read_text())
    val_ids = json.loads(Path(a.val_split).read_text())
    assert len(train_ids) == 4678 and len(val_ids) == 1738
    assert len(set(train_ids)) == len(train_ids) and not set(train_ids) & set(val_ids)
    ids = train_ids[:96]
    split = out / 'calibration_train_ids.json'; write(split, ids)
    model_dir = Path(a.model_dir)
    cfg = yaml_utils.load_yaml(str(model_dir / 'config.yaml'))
    cfg.update(data_dir=a.data_root, root_dir=a.train_split,
               validate_dir=str(split), test_dir=str(split), comm_range=1e6)
    cfg['fusion']['core_method'] = 'intermediatelate'
    model = train_utils.create_model(cfg)
    checkpoint = model_dir / a.checkpoint_name
    saved = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(saved, strict=True); model.cuda().eval()
    counts = None
    if a.sender_thresholds:
        from cosdh_counts import CoSDHCounts
        counts = CoSDHCounts(model, a.sender_thresholds, 8)
    # This calibration never needs actual communication encoding.
    import zlib
    def forbidden(*args, **kwargs): raise AssertionError('Calibration must not use codec')
    zlib.compress = zlib.decompress = forbidden
    dataset = build_dataset(cfg, visualize=False, train=False)
    assert list(dataset.split_info) == ids and len(dataset) == 96
    # Inference preprocessing on training IDs avoids random training augmentation.
    samples = {i: [] for i in range(3)}; cache = {}; handles = []
    def capture(stage):
        def hook(module, inputs, output=None):
            x = output[0] if a.interface == 'post_fusion' else inputs[0]
            assert x.shape[0] == (1 if a.interface == 'post_fusion' else 2) and torch.isfinite(x).all()
            assert stage not in cache, 'Exactly one ego fusion forward required'
            cache[stage] = x[0 if a.interface == 'post_fusion' else 1].detach().cpu().numpy().copy()
        return hook
    if a.interface == 'post_fusion':
        for stage, fusion in enumerate(model.fusion_net):
            handles.append(fusion.register_forward_hook(capture(stage)))
    else:
        for stage, compressor in enumerate(model.naive_compressor_list):
            handles.append(compressor.decoder.register_forward_pre_hook(capture(stage)))
    step_options = (4, 16) if a.interface == 'post_fusion' else (8, 16, 32)
    rows = []; thresholds = {}; audit = {i: {t: [] for t in step_options} for i in range(3)}
    with torch.no_grad():
        for index, frame_id in enumerate(ids):
            cache.clear()
            batch = dataset.collate_batch_test([dataset[index]])
            batch = train_utils.to_device(batch, torch.device('cuda:0'))
            assert batch['ego']['record_len'].tolist() == [2]
            if counts: counts.begin()
            model(batch['ego'])
            if counts: counts.finish()
            assert set(cache) == {0, 1, 2}
            row = dict(frame_id=frame_id, partition='fit' if index < 64 else 'audit', stages=[])
            for stage, x in sorted(cache.items()):
                flat = x.reshape(x.shape[0], -1)
                row['stages'].append(dict(stage=stage, shape=list(x.shape), minimum=float(x.min()),
                    maximum=float(x.max()), negative_fraction=float((x < 0).mean()),
                    nonzero_fraction=float((x != 0).mean())))
                assert (x >= 0).all(), 'Unsigned IF requires verified nonnegative latents'
                if index < 64:
                    # Deterministic spatial sampling; all channels, <=2048 sites/frame.
                    ix = np.linspace(0, flat.shape[1]-1, min(2048, flat.shape[1]), dtype=np.int64)
                    samples[stage].append(flat[:, ix])
                else:
                    scale = np.asarray(thresholds[stage]['range_p999'], dtype=np.float32)[:, None, None]
                    for steps in step_options:
                        theta = scale / steps
                        count = np.minimum(np.floor(x / theta), steps)
                        restored = count * theta
                        audit[stage][steps].append(dict(mse=float(np.mean((restored-x)**2)),
                            energy=float(np.mean(x**2)), mae=float(np.mean(np.abs(restored-x))),
                            clipped_fraction=float((x > scale).mean()),
                            count_nonzero_fraction=float((count > 0).mean()),
                            mean_count=float(count.mean())))
            rows.append(row)
            if index == 63:
                for stage in range(3):
                    fit = np.concatenate(samples[stage], axis=1)
                    ranges = np.maximum(np.quantile(fit, .999, axis=1), 1e-6)
                    thresholds[stage] = dict(range_p999=ranges.tolist(),
                        sampled_values_per_channel=int(fit.shape[1]),
                        minimum_by_channel=fit.min(axis=1).tolist(), maximum_by_channel=fit.max(axis=1).tolist())
                write(out / 'thresholds.json', dict(fit_ids=ids[:64], stages=thresholds,
                    count_formula='min(floor(x/(range_p999/T)),T)', steps=list(step_options), interface=a.interface,
                    note='Analytic constant-current IF count diagnostic; no event selection or SNN encoder yet'))
            write(out / 'status.json', dict(stage='calibrating', completed=index+1, planned=96))
    for handle in handles: handle.remove()
    if counts: counts.close()
    assert all(torch.equal(v.cpu(), saved[k]) for k, v in model.state_dict().items())
    aggregated = {}
    for stage, configs in audit.items():
        aggregated[stage] = {}
        for steps, values in configs.items():
            means = {key: float(np.mean([v[key] for v in values])) for key in values[0]}
            means['normalized_mse'] = means['mse'] / max(means['energy'], 1e-20)
            aggregated[stage][steps] = means
    report = dict(passed=True, samples=96, fit_samples=64, audit_samples=32,
        training_only=True, validation_disjoint=True, model_state_unchanged=True,
        no_ap_selection=True, no_optimization=True, no_wire_calls=True,
        checkpoint_sha256=sha(checkpoint), config_sha256=sha(model_dir/'config.yaml'),
        train_split_sha256=sha(a.train_split), val_split_sha256=sha(a.val_split),
        source_sha256=sha(__file__), thresholds_sha256=sha(out/'thresholds.json'),
        interface=a.interface, sender_steps=8 if counts else None,
        sender_thresholds_sha256=sha(a.sender_thresholds) if counts else None,
        frames=rows, heldout_training_error=aggregated,
        boundary='Representation calibration only; no AP or bytes measured; post_fusion retains original ANN fusion and is not membrane/gate conversion')
    write(out / 'calibration.json', report)
    write(out / 'status.json', dict(stage='complete', completed=96))


if __name__ == '__main__':
    main()
