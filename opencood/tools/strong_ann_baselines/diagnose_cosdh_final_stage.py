"""Training-only intervention audit of the last BEV stage and its receiver path."""
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
    for key in ('repo','model-dir','data-root','train-split','val-split','sender-thresholds','readout-thresholds','front-thresholds','second-thresholds','third-thresholds','output'):
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
    third=CoSDHFrontStage(model,a.third_thresholds,'identity')
    assert (front.stage,second.stage,third.stage)==(0,1,2)
    converters=[counts,readout,membrane,front,second,third]
    def forbidden(*args,**kwargs):raise AssertionError('No codec during diagnosis')
    zlib.compress=zlib.decompress=forbidden
    dataset=build_dataset(cfg,visualize=False,train=False);assert list(dataset.split_info)==ids
    captures={};handles=[]
    def capture(name):
        def hook(module,inputs,output):captures.setdefault(name,[]).append(output.detach().clone())
        return hook
    def capture_input(name):
        def hook(module,inputs):captures.setdefault(name,[]).append(inputs[0].detach().clone())
        return hook
    compressor=model.naive_compressor_list[2]
    handles.append(model.backbone.blocks[2].register_forward_hook(capture('stage_features')))
    handles.append(compressor.encoder.register_forward_hook(capture('compressed_before_fp16')))
    handles.append(compressor.decoder.register_forward_pre_hook(capture_input('decoder_input_after_sender')))
    handles.append(model.fusion_net[2].register_forward_pre_hook(capture_input('decoded_fusion_input')))
    def fusion_capture(module,inputs,output):captures.setdefault('fusion_after_readout',[]).append(output[0].detach().clone())
    handles.append(model.fusion_net[2].register_forward_hook(fusion_capture))
    handles.append(model.cls_head.register_forward_hook(capture('classification_logits')))
    variants=['identity','clip','zero32','half32','zero_stage_features','zero_decoded_features','zero_fused_readout']
    rows=[]
    with torch.no_grad():
        for i,fid in enumerate(ids):
            batch=dataset.collate_batch_test([dataset[i]]);batch=train_utils.to_device(batch,torch.device('cuda:0'))
            assert batch['ego']['record_len'].tolist()==[2]
            baseline=None;frame=dict(frame_id=fid,variants={})
            for variant in variants:
                third.mode='clip' if variant=='clip' else ('count32' if variant in ('zero32','half32') else 'identity')
                third.initial_voltage=.5 if variant=='half32' else 0.
                intervention=None
                if variant=='zero_stage_features':
                    intervention=model.backbone.blocks[2][-1].register_forward_hook(lambda m,x,y:torch.zeros_like(y))
                elif variant=='zero_decoded_features':
                    intervention=compressor.register_forward_hook(lambda m,x,y:torch.zeros_like(y))
                elif variant=='zero_fused_readout':
                    intervention=model.fusion_net[2].register_forward_hook(lambda m,x,y:(torch.zeros_like(y[0]),)+y[1:])
                captures.clear()
                for converter in converters:converter.begin()
                try:result=model(batch['ego'])
                finally:
                    if intervention:intervention.remove()
                routes=[converter.finish() for converter in converters]
                captures['final_cls']=[result['cls_preds'].detach().clone()]
                captures['final_reg']=[result['reg_preds'].detach().clone()]
                if 'dir_preds' in result:captures['final_dir']=[result['dir_preds'].detach().clone()]
                assert len(captures['stage_features'])==len(captures['classification_logits'])==2
                assert all(torch.isfinite(t).all() for values in captures.values() for t in values)
                if baseline is None:baseline={k:v[:] for k,v in captures.items()}
                stats={}
                for key,values in captures.items():
                    assert len(values)==len(baseline[key])
                    stats[key]=[dict(shape=list(t.shape),abs_mean=float(t.abs().mean()),nonzero_fraction=float((t!=0).float().mean()),
                        max_abs_change=float((t-ref).abs().max()),changed=not torch.equal(t,ref)) for t,ref in zip(values,baseline[key])]
                frame['variants'][variant]=dict(tensors=stats,front_calls=routes[3]['calls'],second_calls=routes[4]['calls'],third_calls=routes[5]['calls'])
            rows.append(frame);write(out/'status.json',dict(stage='diagnosing',completed=i+1,planned=32))
    for h in handles:h.remove()
    for converter in converters:converter.close()
    assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items())
    summary={v:{k:dict(frames_changed=sum(any(t['changed'] for t in r['variants'][v]['tensors'][k]) for r in rows),
        max_abs_change=max(t['max_abs_change'] for r in rows for t in r['variants'][v]['tensors'][k])) for k in rows[0]['variants'][v]['tensors']} for v in variants}
    write(out/'diagnosis.json',dict(passed=True,frames=rows,summary=summary,training_only=True,samples=32,model_state_unchanged=True,
        no_optimization=True,no_ap_selection=True,no_wire_calls=True,checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),
        thresholds_sha256={k:sha(getattr(a,k+'_thresholds')) for k in ('sender','readout','front','second','third')},
        boundary='Raw tensor interventions on first32 common training frames, no detection AP or equivalence claim for full validation; no weights changed.'))
    write(out/'status.json',dict(stage='complete',completed=32))
    print(json.dumps(summary))


if __name__=='__main__':main()
