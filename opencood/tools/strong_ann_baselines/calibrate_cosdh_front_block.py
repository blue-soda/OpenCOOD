"""Training-only first residual block activation calibration, no conversion.

The shared ReLU executes twice per block and the encoder executes twice per
ego forward (confidence path and communication path). Both uses are audited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zlib


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(p)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'model-dir', 'data-root', 'train-split', 'val-split', 'sender-thresholds', 'readout-thresholds', 'output'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(a.repo).resolve())); os.chdir(a.repo)
    import random
    import numpy as np
    import torch
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from cosdh_counts import CoSDHCounts
    from cosdh_readout import CoSDHReadout
    from cosdh_membrane import CoSDHMembrane
    random.seed(303); np.random.seed(303); torch.manual_seed(303); torch.cuda.manual_seed_all(303)
    torch.set_num_threads(1); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.25)
    train = json.loads(Path(a.train_split).read_text()); val = json.loads(Path(a.val_split).read_text())
    assert len(train)==len(set(train))==4678 and len(val)==1738 and not set(train)&set(val)
    ids=train[:96]; split=out/'calibration_train_ids.json'; write(split, ids)
    model_dir=Path(a.model_dir); checkpoint=model_dir/'net_epoch3.pth'
    cfg=yaml_utils.load_yaml(str(model_dir/'config.yaml'))
    cfg.update(data_dir=a.data_root, root_dir=a.train_split, validate_dir=str(split), test_dir=str(split), comm_range=1e6)
    cfg['fusion']['core_method']='intermediatelate'
    model=train_utils.create_model(cfg); original=torch.load(str(checkpoint),map_location='cpu')
    model.load_state_dict(original,strict=True); model.cuda().eval()
    counts=CoSDHCounts(model,a.sender_thresholds,8)
    readout=CoSDHReadout(model,a.readout_thresholds,'count16')
    membrane=CoSDHMembrane(model,a.readout_thresholds,'local',16)
    first=model.backbone.resnet.layer0[0]
    assert type(first).__name__=='BasicBlock' and type(first.relu).__name__=='ReLU'
    def forbidden(*args,**kwargs): raise AssertionError('No codec during calibration')
    zlib.compress=zlib.decompress=forbidden
    dataset=build_dataset(cfg,visualize=False,train=False); assert list(dataset.split_info)==ids
    calls=[]
    def capture(module,inputs,output):
        x=output.detach().cpu().numpy().copy()
        assert x.ndim==4 and x.shape[0]==2 and np.isfinite(x).all() and (x>=0).all()
        calls.append(x)
    handle=first.relu.register_forward_hook(capture)
    samples={i:[] for i in (0,1)}; scales={}; rows=[]; errors={i:{t:[] for t in (16,32)} for i in (0,1)}
    with torch.no_grad():
        for index,fid in enumerate(ids):
            calls.clear(); counts.begin(); readout.begin(); membrane.begin()
            batch=dataset.collate_batch_test([dataset[index]]); batch=train_utils.to_device(batch,torch.device('cuda:0'))
            assert batch['ego']['record_len'].tolist()==[2]
            model(batch['ego']); counts.finish(); readout.finish(); membrane.finish()
            assert len(calls)==4 and np.array_equal(calls[0],calls[2]) and np.array_equal(calls[1],calls[3])
            row=dict(frame_id=fid,partition='fit' if index<64 else 'audit',relu_calls=4,two_encoder_paths_bitwise_equal=True,sites=[])
            for phase in (0,1):
                x=calls[phase]; flat=x.transpose(1,0,2,3).reshape(x.shape[1],-1)
                row['sites'].append(dict(phase=phase,shape=list(x.shape),minimum=float(x.min()),maximum=float(x.max()),nonzero_fraction=float((x>0).mean())))
                if index<64:
                    ix=np.linspace(0,flat.shape[1]-1,min(2048,flat.shape[1]),dtype=np.int64); samples[phase].append(flat[:,ix])
                else:
                    scale=np.asarray(scales[phase]['range_p999'],dtype=np.float32)[None,:,None,None]
                    for steps in (16,32):
                        q=np.floor(x/(scale/steps)).clip(0,steps); restored=q*(scale/steps)
                        errors[phase][steps].append(dict(mse=float(np.mean((restored-x)**2)),energy=float(np.mean(x*x)),clipped_fraction=float((x>scale).mean()),count_nonzero_fraction=float((q>0).mean())))
            rows.append(row)
            if index==63:
                for phase in (0,1):
                    fit=np.concatenate(samples[phase],axis=1); scales[phase]=dict(range_p999=np.maximum(np.quantile(fit,.999,axis=1),1e-6).tolist(),sampled_values_per_channel=int(fit.shape[1]))
                write(out/'thresholds.json',dict(interface='backbone.resnet.layer0.0.relu',phases=scales,fit_ids=ids[:64],shared_agent_thresholds=True,expected_calls_per_forward=4,steps=[16,32],checkpoint_sha256=sha(checkpoint)))
            write(out/'status.json',dict(stage='calibrating',completed=index+1,planned=96))
    handle.remove(); counts.close(); readout.close(); membrane.close()
    assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items())
    means={}
    for phase in errors:
        means[phase]={}
        for steps,values in errors[phase].items():
            m={k:float(np.mean([v[k] for v in values])) for k in values[0]};m['normalized_mse']=m['mse']/max(m['energy'],1e-20);means[phase][steps]=m
    write(out/'calibration.json',dict(passed=True,samples=96,fit_samples=64,audit_samples=32,training_only=True,validation_disjoint=True,model_state_unchanged=True,no_optimization=True,no_ap_selection=True,no_wire_calls=True,
        checkpoint_sha256=sha(checkpoint),config_sha256=sha(model_dir/'config.yaml'),thresholds_sha256=sha(out/'thresholds.json'),source_sha256=sha(__file__),train_split_sha256=sha(a.train_split),val_split_sha256=sha(a.val_split),
        frames=rows,heldout_training_error=means,boundary='Calibration only, first residual block has NOT been converted. Both agents share thresholds; both confidence and communication encoder invocations must later use the same converted activations. Skip remains continuous until separately changed.'))
    write(out/'status.json',dict(stage='complete',completed=96))


if __name__=='__main__': main()
