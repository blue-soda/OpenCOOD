"""Train-prefix, teacher-free inference diagnostics for V2 state prediction.

Compare one-step stale history with a cheap current-evidence delta, always under
the original dense history. This does not implement recursive selective inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_unified_bev import UnifiedBEVDetector
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--frames',type=int,default=256);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:run(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


def run(a,out):
    for k,v in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0).items():setattr(a,k,v)
    teacher,cfg,_,provenance=setup(a)
    source=Path(a.checkpoint);sha=digest(source)
    assert json.loads((source.parent/'report.json').read_text())['checkpoint_sha256']==sha
    saved=torch.load(str(source),map_location='cpu')
    assert saved['contract']['memory_policy']=='recurrent'
    assert saved['contract']['data_contract']==provenance['contract']
    student=UnifiedBEVDetector(teacher,cfg,'v2','recurrent').cuda().eval()
    student.load_state_dict(saved['model'],strict=True)
    runtime=json.loads((Path(a.root)/'full_gru/effective_runtime.json').read_text())['arguments']
    rows=json.loads(Path(runtime['manifest']).read_text())['train'][:a.frames]
    captured={}
    def packed_hook(module,args):captured['packed']=args[0]
    def net_hook(module,args,result):captured['net']=result
    hooks=[student.core.update.register_forward_pre_hook(packed_hook),student.core.update.net.register_forward_hook(net_hook)]
    state=previous=None;records=[];_,stream=loader(rows,cfg,2,303)
    try:
        with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
            for index,(row,sample) in enumerate(zip(rows,stream)):
                assert sample['meta']['frame']==row['frame']
                data=to_device(sample['ego'],torch.device('cuda'))
                _,state,info=student.step(data,state,sample['meta'])
                packed=captured['packed'];evidence=packed[:,:256];old=packed[:,256:512]
                record=dict(frame=row['frame'],segment=row['segment'],reset=info['reset'],supervised=row['supervised'])
                if previous is not None and info['reset'] is None:
                    aligned,valid=warp_bev(previous['evidence'],previous['pose'],sample['meta']['pose'],student.bounds)
                    valid=valid & packed[:,-1:].bool()
                    # Bias cancels; weight is the model's existing cheap current path.
                    current=student.core.update.current(evidence)
                    delta=current-student.core.update.current(aligned)
                    corrected=old+delta;target=state['value']
                    retain=captured['net'][:,256:].sigmoid()*packed[:,-1:]
                    scopes={'valid':valid}
                    if row['supervised']:
                        positive=data['label_dict']['pos_equal_one']
                        assert positive.shape[1:3]==target.shape[-2:]
                        foreground=positive.bool().any(-1).unsqueeze(1)
                        scopes.update(foreground=valid & foreground,background=valid & ~foreground)
                    for name,mask in scopes.items():
                        if not bool(mask.any()):continue
                        select=mask.expand_as(target)
                        rms=lambda x:float(x[select].square().mean().sqrt())
                        stale=rms(old-target);delta_error=rms(corrected-target)
                        record[name]=dict(cells=int(mask.sum()),stale_rms=stale,delta_rms=delta_error,
                            delta_to_stale=delta_error/max(stale,1e-12),target_rms=rms(target),current_rms=rms(current),
                            residual_rms=rms(captured['net'][:,:256]),retain_mean=float(retain[select].mean()),
                            retain_lt01=float((retain[select]<.01).float().mean()),retain_gt99=float((retain[select]>.99).float().mean()))
                previous=dict(evidence=evidence.detach(),pose=sample['meta']['pose'])
                records.append(record);log.write(json.dumps(record)+'\n')
                if (index+1)%64==0:write_json(out/'status.json',dict(status='RUNNING',frames=index+1))
    finally:
        for hook in hooks:hook.remove()
    assert len(records)==len(rows)
    summaries={}
    for scope in ['valid','foreground','background']:
        values=[r[scope] for r in records if scope in r]
        if values:
            summaries[scope]=dict(frames=len(values),mean_frame_delta_to_stale=float(np.mean([x['delta_to_stale'] for x in values])),
                median_frame_delta_to_stale=float(np.median([x['delta_to_stale'] for x in values])),
                fraction_frames_improved=float(np.mean([x['delta_rms']<x['stale_rms'] for x in values])),
                mean_retain=float(np.mean([x['retain_mean'] for x in values])),
                mean_retain_lt01=float(np.mean([x['retain_lt01'] for x in values])),
                mean_retain_gt99=float(np.mean([x['retain_gt99'] for x in values])))
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint=str(source),checkpoint_sha256=sha,split='train',
        frames=len(records),seed=303,summary=summaries,
        caveats=['One-step dense-history diagnostic, not a recursive selective implementation, latency test or AP result.',
                 'Raw state RMS is sensitive to feature scale and is not equivalent to detection accuracy.',
                 'Untrained fixed unit delta gain; no validation-label or threshold selection.',
                 'Foreground means positive anchor cells on labelled training scans; unlabelled cells are not assumed background.']))
    write_json(out/'status.json',dict(status='COMPLETED'))


if __name__=='__main__':main()
