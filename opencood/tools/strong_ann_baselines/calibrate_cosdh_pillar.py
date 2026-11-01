"""Training-only shared PFN thresholds, before point max, fixed BEV IFs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zlib


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','model-dir','data-root','train-split','val-split','sender-thresholds','readout-thresholds','front-thresholds','second-thresholds','third-thresholds','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(a.repo).resolve()));os.chdir(a.repo)
    import random
    import numpy as np
    import torch
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from cosdh_counts import CoSDHCounts
    from cosdh_readout import CoSDHReadout
    from cosdh_membrane import CoSDHMembrane
    from cosdh_front_stage import CoSDHFrontStage
    from cosdh_pillar import activate_pillar
    random.seed(303);np.random.seed(303);torch.manual_seed(303);torch.cuda.manual_seed_all(303)
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.4)
    train=json.loads(Path(a.train_split).read_text());val=json.loads(Path(a.val_split).read_text())
    assert len(train)==4678 and len(val)==1738 and not set(train)&set(val)
    ids=train[:96];split=out/'train_ids.json';write(split,ids)
    model_dir=Path(a.model_dir);checkpoint=model_dir/'net_epoch3.pth'
    cfg=yaml_utils.load_yaml(str(model_dir/'config.yaml'))
    cfg.update(data_dir=a.data_root,root_dir=a.train_split,validate_dir=str(split),test_dir=str(split),comm_range=1e6);cfg['fusion']['core_method']='intermediatelate'
    model=train_utils.create_model(cfg);original=torch.load(str(checkpoint),map_location='cpu');model.load_state_dict(original,strict=True);model.cuda().eval()
    converters=[CoSDHCounts(model,a.sender_thresholds,8),CoSDHReadout(model,a.readout_thresholds,'count16'),CoSDHMembrane(model,a.readout_thresholds,'local',16),
        CoSDHFrontStage(model,a.front_thresholds,'count32'),CoSDHFrontStage(model,a.second_thresholds,'count32',initial_voltage=.5),CoSDHFrontStage(model,a.third_thresholds,'count32')]
    assert len(model.pillar_vfe.pfn_layers)==1 and model.pillar_vfe.pfn_layers[0].last_vfe
    def forbidden(*args,**kwargs):raise AssertionError('No codec during calibration')
    zlib.compress=zlib.decompress=forbidden
    dataset=build_dataset(cfg,visualize=False,train=False);assert list(dataset.split_info)==ids
    captures=[]
    def capture(module,inputs,output):captures.append(output.detach().clone())
    handle=model.pillar_vfe.pfn_layers[0].norm.register_forward_hook(capture)
    samples=[];rows=[];scale=None
    with torch.no_grad():
        for index,fid in enumerate(ids):
            captures.clear()
            for c in converters:c.begin()
            batch=dataset.collate_batch_test([dataset[index]]);batch=train_utils.to_device(batch,torch.device('cuda:0'));assert batch['ego']['record_len'].tolist()==[2]
            agent_ids=batch['ego']['processed_lidar']['voxel_coords'][:,0].long();assert sorted(agent_ids.unique().tolist())==[0,1]
            model(batch['ego']);routes=[c.finish() for c in converters]
            assert len(captures)==1
            x=captures[0].clamp(min=0);assert x.ndim==3 and x.shape[1]==64 and x.shape[0]==agent_ids.shape[0] and torch.isfinite(x).all()
            row=dict(frame_id=fid,partition='fit' if index<64 else 'audit',pfn_calls=1,shape=list(x.shape),agents=[],BEV_routes=routes[3:])
            for agent in (0,1):
                y=x[agent_ids==agent];flat=y.permute(1,0,2).reshape(64,-1)
                info=dict(agent=agent,pillars=y.shape[0],minimum=float(y.min()),maximum=float(y.max()),nonzero_fraction=float((y>0).float().mean()))
                if index<64:
                    ix=torch.linspace(0,flat.shape[1]-1,min(2048,flat.shape[1]),device=x.device).long();samples.append(flat[:,ix].cpu().numpy())
                else:
                    info['errors']={}
                    for mode,v0,label in [('clip',0.,'clip'),('count32',0.,'zero32'),('count32',.5,'half32')]:
                        q=activate_pillar(y,scale,mode,v0)
                        mse=float(((q-y)**2).mean());energy=float((y*y).mean())
                        info['errors'][label]=dict(mse=mse,energy=energy,normalized_mse=mse/max(energy,1e-20),nonzero_fraction=float((q!=0).float().mean()))
                row['agents'].append(info)
            rows.append(row)
            if index==63:
                values=np.concatenate(samples,axis=1);ranges=np.maximum(np.quantile(values,.999,axis=1),1e-6).tolist()
                scale=torch.tensor(ranges,device=x.device,dtype=x.dtype).reshape(1,64,1)
                write(out/'thresholds.json',dict(interface='pillar_vfe.pfn_layers.0.norm',range_p999=ranges,fit_ids=ids[:64],before_point_max=True,shared_agent_thresholds=True,
                    sampled_values_per_channel=values.shape[1],sampling='up to2048 evenly spaced positions per agent per frame; includes padded positions used by source PFN',checkpoint_sha256=sha(checkpoint)))
            write(out/'status.json',dict(stage='calibrating',completed=index+1,planned=96))
    handle.remove()
    for c in converters:c.close()
    assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items())
    errors={}
    for label in ('clip','zero32','half32'):
        entries=[agent['errors'][label] for r in rows[64:] for agent in r['agents']]
        mse=float(np.mean([e['mse'] for e in entries]));energy=float(np.mean([e['energy'] for e in entries]));errors[label]=dict(mse=mse,energy=energy,normalized_mse=mse/max(energy,1e-20))
    write(out/'calibration.json',dict(passed=True,samples=96,fit_samples=64,audit_samples=32,training_only=True,validation_disjoint=True,model_state_unchanged=True,no_optimization=True,no_ap_selection=True,no_wire_calls=True,
        checkpoint_sha256=sha(checkpoint),thresholds_sha256=sha(out/'thresholds.json'),source_sha256=sha(__file__),train_split_sha256=sha(a.train_split),val_split_sha256=sha(a.val_split),
        fixed_thresholds_sha256={k:sha(getattr(a,k+'_thresholds')) for k in ('sender','readout','front','second','third')},frames=rows,heldout_training_error=errors,
        boundary='PFN stays continuous during training-only calibration; all19 BEV IF activations retained. Padded point slots are included because source does not remask after BN. One shared PFN output feeds both BEV paths.'))
    write(out/'status.json',dict(stage='complete',completed=96))


if __name__=='__main__':main()
