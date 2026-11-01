"""Training-only bitwise equivalence audit of the single-pass geometric backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zlib


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','model-dir','data-root','train-split','val-split','sender-thresholds','readout-thresholds','front-thresholds','second-thresholds','third-thresholds','pillar-thresholds','output'):
        parser.add_argument('--'+key,required=True)
    a=parser.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
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
    random.seed(303);np.random.seed(303);torch.manual_seed(303);torch.cuda.manual_seed_all(303)
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.4)
    train=json.loads(Path(a.train_split).read_text());val=json.loads(Path(a.val_split).read_text())
    assert len(train)==4678 and len(val)==1738 and not set(train)&set(val)
    ids=train[:32];split=out/'train_ids.json';write(split,ids)
    model_dir=Path(a.model_dir);checkpoint=model_dir/'net_epoch3.pth'
    cfg=yaml_utils.load_yaml(str(model_dir/'config.yaml'))
    cfg.update(data_dir=a.data_root,root_dir=a.train_split,validate_dir=str(split),test_dir=str(split),comm_range=1e6)
    cfg['fusion']['core_method']='intermediatelate'
    model=train_utils.create_model(cfg);original=torch.load(str(checkpoint),map_location='cpu')
    model.load_state_dict(original,strict=True);model.cuda().eval()
    counts=CoSDHCounts(model,a.sender_thresholds,8)
    readout=CoSDHReadout(model,a.readout_thresholds,'count16')
    membrane=CoSDHMembrane(model,a.readout_thresholds,'local',16)
    front=CoSDHFrontStage(model,a.front_thresholds,'count32')
    second=CoSDHFrontStage(model,a.second_thresholds,'count32',initial_voltage=.5)
    third=CoSDHFrontStage(model,a.third_thresholds,'count32')
    from cosdh_pillar import CoSDHPillar
    from cosdh_selection import CoSDHSelection
    pillar=CoSDHPillar(model,a.pillar_thresholds,'count32',initial_voltage=.5)
    selection=None
    from cosdh_geometry import GeometryBackend
    assert (front.stage,second.stage,third.stage)==(0,1,2)
    converters=[counts,readout,membrane,front,second,third,pillar]
    def forbidden(*args,**kwargs):raise AssertionError('No codec during diagnosis')
    zlib.compress=zlib.decompress=forbidden
    dataset=build_dataset(cfg,visualize=False,train=False);assert list(dataset.split_info)==ids
    rows=[]
    with torch.no_grad():
        for i,fid in enumerate(ids):
            batch=dataset.collate_batch_test([dataset[i]]);batch=train_utils.to_device(batch,torch.device('cuda:0'))
            assert batch['ego']['record_len'].tolist()==[2]
            for c in (front,second,third,pillar):c.encoder_paths=2
            selection=CoSDHSelection(model,'geometry_only')
            for c in converters+[selection]:c.begin()
            legacy=model(batch['ego']);legacy={k:v.detach().clone() for k,v in legacy.items()}
            old_routes=[c.finish() for c in converters];old_mask=selection.finish();selection.close()
            for c in (front,second,third,pillar):c.encoder_paths=1
            geometry=GeometryBackend(model)
            for c in converters+[geometry]:c.begin()
            try:new=model(batch['ego'])
            finally:geometry.close()
            new_routes=[c.finish() for c in converters];new_geometry=geometry.finish()
            assert set(legacy)==set(new)
            for k in legacy:
                assert torch.isfinite(new[k]).all() and torch.equal(legacy[k],new[k]),(fid,k,float((legacy[k]-new[k]).abs().max()))
            assert [r['road_retained_cells'] for r in old_mask['stages']]==[r['road_retained_cells'] for r in new_geometry['stages']]
            assert [r['calls'] for r in old_routes[3:6]]==[[2]*4,[2]*6,[2]*9]
            assert [r['calls'] for r in new_routes[3:6]]==[[1]*4,[1]*6,[1]*9]
            rows.append(dict(frame_id=fid,raw_outputs_bitwise_equal=True,head_keys=list(new),old_BEV_calls=[r['calls'] for r in old_routes[3:6]],new_BEV_calls=[r['calls'] for r in new_routes[3:6]],
                old_selection=old_mask,new_geometry=new_geometry,PFN_calls=new_routes[-1]['calls']))
            write(out/'status.json',dict(stage='diagnosing',completed=i+1,planned=32))
    for c in converters:c.close()
    assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items())
    write(out/'diagnosis.json',dict(passed=True,samples=32,training_only=True,frames=rows,model_state_unchanged=True,no_optimization=True,no_ap_selection=True,no_wire_calls=True,
        all_raw_outputs_bitwise_equal=True,checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),
        thresholds_sha256={k:sha(getattr(a,k+'_thresholds')) for k in ('sender','readout','front','second','third','pillar')},
        boundary='32 training-frame raw cls/reg/dir equality; full validation must still be verified. Single encoder, no confidence backbone forward/Communication/request scatter, local membrane and IF readout retained.'))
    write(out/'status.json',dict(stage='complete',completed=32));print('32 training frames bitwise equivalent, single encoder and removed-route guards passed')


if __name__=='__main__':main()
