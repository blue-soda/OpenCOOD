"""Training-only confidence/request dependency and invariance audit."""
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
    selection=CoSDHSelection(model,'identity')
    assert (front.stage,second.stage,third.stage)==(0,1,2)
    converters=[counts,readout,membrane,front,second,third,pillar,selection]
    def forbidden(*args,**kwargs):raise AssertionError('No codec during diagnosis')
    zlib.compress=zlib.decompress=forbidden
    dataset=build_dataset(cfg,visualize=False,train=False);assert list(dataset.split_info)==ids
    masks=[];handles=[]
    def capture(module,inputs,output):masks.append(output[0].detach().clone())
    for fusion in model.fusion_net:handles.append(fusion.naive_communication.register_forward_hook(capture))
    variants=['identity','request_only','confidence_only','geometry_only','request_only_perturb_confidence','confidence_only_perturb_request','geometry_only_perturb_both']
    rows=[]
    with torch.no_grad():
        for i,fid in enumerate(ids):
            batch=dataset.collate_batch_test([dataset[i]]);batch=train_utils.to_device(batch,torch.device('cuda:0'))
            assert batch['ego']['record_len'].tolist()==[2]
            baseline=None;references={};frame=dict(frame_id=fid,variants={})
            for variant in variants:
                mode=variant.split('_perturb_')[0];selection.mode=mode;interventions=[]
                if '_perturb_' in variant:
                    what=variant.split('_perturb_')[1]
                    def perturb(module,inputs):
                        conf,B,warp,req=inputs
                        if what in ('confidence','both'):conf=[torch.full_like(v,-100.) for v in conf]
                        if what in ('request','both'):req=torch.zeros_like(req)
                        return conf,B,warp,req
                    for fusion in model.fusion_net:interventions.append(fusion.naive_communication.register_forward_pre_hook(perturb))
                masks.clear()
                for c in converters:c.begin()
                try:result=model(batch['ego'])
                finally:
                    for h in interventions:h.remove()
                routes=[c.finish() for c in converters];assert len(masks)==3
                outputs={k:v.detach().clone() for k,v in result.items() if k in ('cls_preds','reg_preds','dir_preds')}
                assert all(torch.isfinite(v).all() for v in outputs.values())
                current=(masks[:],outputs)
                if baseline is None:baseline=current
                if '_perturb_' not in variant:references[mode]=current
                else:
                    ref=references[mode]
                    assert all(torch.equal(v,w) for v,w in zip(masks,ref[0]))
                    assert all(torch.equal(v,ref[1][k]) for k,v in outputs.items()),variant
                frame['variants'][variant]=dict(mask_changed_cells=[int((v!=w).sum()) for v,w in zip(masks,baseline[0])],
                    mask_road_cells=[int(v[1].sum()) for v in masks],
                    output_max_abs_change={k:float((v-baseline[1][k]).abs().max()) for k,v in outputs.items()},
                    removed_dependency_invariance_passed=('_perturb_' in variant),selection=routes[-1],
                    PFN_calls=routes[-2]['calls'],BEV_calls=[r['calls'] for r in routes[3:6]])
            rows.append(frame);write(out/'status.json',dict(stage='diagnosing',completed=i+1,planned=32))
    for h in handles:h.remove()
    for c in converters:c.close()
    assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items())
    summary={v:dict(mask_changed_frames=sum(any(r['variants'][v]['mask_changed_cells']) for r in rows),
        raw_output_changed_frames=sum(any(x>0 for x in r['variants'][v]['output_max_abs_change'].values()) for r in rows),
        max_raw_cls_change=max(r['variants'][v]['output_max_abs_change']['cls_preds'] for r in rows)) for v in variants}
    write(out/'diagnosis.json',dict(passed=True,samples=32,training_only=True,frames=rows,summary=summary,model_state_unchanged=True,
        no_optimization=True,no_ap_selection=True,no_wire_calls=True,removed_dependency_invariance_all_frames_passed=True,
        checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),
        thresholds_sha256={k:sha(getattr(a,k+'_thresholds')) for k in ('sender','readout','front','second','third','pillar')},
        boundary='Actual mask/output differences may be zero: redundancy is a valid hypothesis. Removed confidence/request dependence must pass counterfactual input invariance on32 train frames. No full-validation AP claim.'))
    write(out/'status.json',dict(stage='complete',completed=32));print(json.dumps(summary))


if __name__=='__main__':main()
