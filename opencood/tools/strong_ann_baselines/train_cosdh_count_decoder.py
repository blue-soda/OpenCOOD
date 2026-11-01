"""Three-epoch matched decoder adaptation for ANN / IF8 / IF32 CoSDH.

Frozen encoders, fusion, detection head and all BN tensors; inference masks and
FP16 latent preprocessing on the common training split. No wire or validation
selection. --smoke performs two discarded real steps before formal admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zlib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'model-dir', 'data-root', 'train-split', 'val-split', 'thresholds', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--steps', type=int, choices=(0, 8, 32), required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--checkpoint-name', default='net_epoch11.pth')
    p.add_argument('--readout-thresholds')
    p.add_argument('--readout-mode', choices=('identity', 'clip', 'count4', 'count16'))
    p.add_argument('--membrane-mode', choices=('identity', 'fixed', 'local'))
    a = p.parse_args()
    if a.membrane_mode and a.readout_mode not in ('count4', 'count16'):
        p.error('Membrane adaptation requires calibrated count readout')
    if bool(a.readout_thresholds) != bool(a.readout_mode):
        p.error('Readout thresholds and mode required together')
    if a.readout_mode and a.steps != 8:
        p.error('Readout adaptation requires fixed IF8 sender')
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(a.repo).resolve())); os.chdir(a.repo)
    import numpy as np
    import torch
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from cosdh_counts import CoSDHCounts

    random.seed(303); np.random.seed(303); torch.manual_seed(303); torch.cuda.manual_seed_all(303)
    torch.set_num_threads(1); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False; torch.cuda.set_per_process_memory_fraction(.6)
    train_ids = json.loads(Path(a.train_split).read_text()); val_ids = json.loads(Path(a.val_split).read_text())
    assert len(train_ids) == len(set(train_ids)) == 4678 and len(val_ids) == 1738
    assert not set(train_ids) & set(val_ids)
    model_dir = Path(a.model_dir); cfg = yaml_utils.load_yaml(str(model_dir/'config.yaml'))
    cfg.update(data_dir=a.data_root, root_dir=a.train_split, validate_dir=a.train_split,
               test_dir=a.train_split, comm_range=1e6)
    cfg['fusion']['core_method'] = 'intermediatelate'
    checkpoint = model_dir/a.checkpoint_name; original = torch.load(str(checkpoint), map_location='cpu')
    model = train_utils.create_model(cfg); model.load_state_dict(original, strict=True); model.cuda().eval()
    for param in model.parameters(): param.requires_grad_(False)
    names = []
    for name, module in model.named_modules():
        if name.startswith('naive_compressor_list.') and '.decoder.' in name and isinstance(module, torch.nn.Conv2d):
            for suffix, param in module.named_parameters(recurse=False):
                param.requires_grad_(True); names.append(name+'.'+suffix)
    assert len(names) == 12, names
    parameters = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-4)
    criterion = train_utils.create_loss(cfg)
    counts = CoSDHCounts(model, a.thresholds, a.steps) if a.steps else None
    readout = None
    if a.readout_mode:
        from cosdh_readout import CoSDHReadout
        readout = CoSDHReadout(model, a.readout_thresholds, a.readout_mode, surrogate=True)
    membrane = None
    if a.membrane_mode:
        from cosdh_membrane import CoSDHMembrane
        membrane = CoSDHMembrane(model, a.readout_thresholds, a.membrane_mode, int(a.readout_mode[5:]))
    dataset = build_dataset(cfg, visualize=False, train=False)
    assert list(dataset.split_info) == train_ids
    contract = dict(epochs=3, batch_size=1, seed=303, lr=1e-5, weight_decay=1e-4, optimizer='AdamW',
        grad_clip=10., steps=a.steps, trainable_names=names, trainable_parameters=sum(v.numel() for v in parameters),
        scope='Shared local/remote decoder convolutions only; all other tensors and all BN frozen; model.eval inference masks',
        preprocessing='Common training split with inference preprocessing, no random augmentation; shuffled each epoch',
        selection='Last epoch3 only; no intermediate AP selection; pre-adaptation references retained',
        checkpoint_sha256=sha(checkpoint), train_split_sha256=sha(a.train_split), val_split_sha256=sha(a.val_split),
        thresholds_sha256=sha(a.thresholds), source_sha256=sha(__file__), no_wire=True,
        membrane_mode=a.membrane_mode, membrane_parameters=0,
        readout_mode=a.readout_mode, readout_thresholds_sha256=sha(a.readout_thresholds) if readout else None,
        readout_surrogate='Unit derivative on [0,scale], zero outside; exact IF forward' if readout and a.readout_mode.startswith('count') else None)
    write(root/'contract.json', contract)
    def forbidden(*args, **kwargs): raise AssertionError('Training must not compress/decompress')
    zlib.compress = zlib.decompress = forbidden
    history = []; updates = 0; start = time.time()
    for epoch in range(1, 2 if a.smoke else 4):
        order = np.random.RandomState(303+epoch).permutation(len(dataset))
        if a.smoke: order = order[:2]
        order_hash = hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
        loss_sum = 0.
        for batch_index, index in enumerate(order, 1):
            batch = dataset.collate_batch_test([dataset[int(index)]])
            batch = train_utils.to_device(batch, torch.device('cuda:0'))
            assert batch['ego']['record_len'].tolist() == [2]
            if counts: counts.begin()
            if readout: readout.begin()
            if membrane: membrane.begin()
            optimizer.zero_grad(set_to_none=True)
            output = model(batch['ego'])
            if counts: counts.finish()
            if readout: readout.finish()
            membrane_diagnostic = membrane.finish() if membrane else None
            loss = criterion(output, batch['ego']['label_dict'])
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
            assert any(v.grad is not None and torch.count_nonzero(v.grad) > 0 for v in parameters)
            optimizer.step(); updates += 1; loss_sum += float(loss.detach())
            if batch_index == 1 or batch_index % 25 == 0 or a.smoke:
                row = dict(state='training', epoch=epoch, batch=batch_index, batches=len(order),
                    optimizer_steps=updates, loss=float(loss.detach()), gradient_norm=float(norm),
                    elapsed_seconds=time.time()-start, order_sha256=order_hash)
                if membrane_diagnostic: row['membrane'] = membrane_diagnostic
                write(root/'progress.json', row)
                with (root/'steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            del output, loss, batch
        state = model.state_dict()
        changed = [k for k,v in state.items() if not torch.equal(v.detach().cpu(), original[k])]
        assert changed and set(changed).issubset(names), changed
        assert all(torch.isfinite(v).all() for v in parameters)
        history.append(dict(epoch=epoch, samples=len(order), loss=loss_sum/len(order), order_sha256=order_hash,
                            changed_names=changed, frozen_tensors_unchanged=True))
        write(root/'history.json', history)
        if a.smoke:
            write(root/'smoke.json', dict(passed=True, optimizer_steps=updates, discarded=True,
                frozen_tensors_unchanged=True, actual_parameter_updates=True, no_compression=True,
                history=history, contract=contract))
            return
        assert len(order) == 4678
        assert shutil.disk_usage(str(root)).free > 1024**3
        tmp = root/'last.tmp'
        torch.save(dict(model=state, optimizer=optimizer.state_dict(), epoch=epoch, history=history,
            contract=contract, rng=dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())), str(tmp))
        os.replace(str(tmp), str(root/'last.pt'))
    dest = root/'final_model'; dest.mkdir()
    shutil.copyfile(str(model_dir/'config.yaml'), str(dest/'config.yaml'))
    torch.save(model.state_dict(), str(dest/'net_epoch3.pth'))
    write(root/'completed.json', dict(epochs=3, optimizer_steps=updates, no_ap_selection=True,
        frozen_tensors_unchanged=True, checkpoint_sha256=sha(dest/'net_epoch3.pth'),
        config_sha256=sha(dest/'config.yaml'), history=history))
    write(root/'progress.json', dict(state='complete', epoch=3, batch=4678, optimizer_steps=updates))


if __name__ == '__main__':
    main()
