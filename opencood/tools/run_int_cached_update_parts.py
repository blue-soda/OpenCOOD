"""Fixed-checkpoint paired pilot: cache expensive parts, keep current evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_unified_bev import UnifiedBEVDetector
from opencood.models.sub_modules.int_cached_update_parts import CachedUpdateParts
from opencood.models.sub_modules.int_unified_gather import gathered_update
from opencood.tools.run_int_unified_bev import engineering
from opencood.tools.run_int_cache_repair import setup,evaluate
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=512)
    p.add_argument('--tile',type=int,default=16);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:run(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


def run(a,out):
    for k,v in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0,
        oracle_stage0=False,oracle_index=0,support_profile=False,zero_uncovered=False).items():setattr(a,k,v)
    source=Path(a.checkpoint);sha=digest(source)
    assert json.loads((source.parent/'report.json').read_text())['checkpoint_sha256']==sha
    saved=torch.load(str(source),map_location='cpu')
    versions={'unified_bev_dense_v2':'v2','unified_bev_dense_v2n':'v2n'}
    assert saved['contract']['version'] in versions and saved['contract']['memory_policy']=='recurrent'
    variant=versions[saved['contract']['version']]
    write_json(out/'status.json',dict(status='RUNNING',phase='engineering'))
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov)
    assert saved['contract']['data_contract']==prov['contract']
    student=UnifiedBEVDetector(teacher,cfg,variant,'recurrent').cuda().eval()
    student.load_state_dict(saved['model'],strict=True)
    for parameter in student.parameters():parameter.requires_grad_(False)
    a.engineering=str(out/'engineering.json');engineering(student,teacher,cfg,rows,out)
    wrapped=CachedUpdateParts(student,period=1);plain=cache=None;errors=[];captured=[]
    hook=student.core.update.register_forward_pre_hook(lambda m,args:captured.append(args[0]) if not captured else None)
    _,stream=loader(rows[:8],cfg,2,303)
    try:
        with torch.no_grad():
            for sample in stream:
                data=to_device(sample['ego'],torch.device('cuda'))
                expected,plain,_=student.step(data,plain,sample['meta'])
                got,cache,_=wrapped.step(data,cache,sample['meta'])
                error=max(float((got[k]-expected[k]).abs().max()) for k in got);errors.append(error)
                assert all(torch.equal(got[k],expected[k]) for k in got)
            x=captured[0];expected=student.core.update.net(x);mask=torch.zeros_like(x[:,:1],dtype=torch.bool)
            mask[...,0:16,0:16]=True;mask[...,-16:,-16:]=True
            got=gathered_update(student.core.update,x,mask,cached_parts=torch.zeros_like(expected))
            selected=mask.expand_as(got);err=(got-expected).abs()[selected]
            assert torch.allclose(got[selected],expected[selected],atol=5e-4,rtol=1e-4)
            assert torch.equal(got[~selected],torch.zeros_like(got[~selected]))
            write_json(out/'parts_engineering.json',dict(status='PASSED',period1_prediction_max_errors=errors,
                parts_max_abs_error=float(err.max()),checkpoint_sha256=sha))
    finally:hook.remove()
    write_json(out/'experiment_contract.json',dict(checkpoint=str(source),checkpoint_sha256=sha,source_epoch=saved['epoch'],
        source_training_contract=saved['contract'],variant=variant,periods=[2,3],max_age=.25,tile=a.tile,
        rule='Reuse residual/gate logits, current projection and recurrent value updated everywhere; new-view/age cells use gathered net; >=.8 unique halo support falls back dense.',
        calibration='No fitted selector or threshold. Periods fixed before validation.',validation_scans=a.frames or len(rows)))
    reports={}
    for name,period,mode in [('dense_before',None,None),('period2',2,'gather'),('masked_period3',3,'masked_dense'),
                              ('period3',3,'gather'),('dense_after',None,None)]:
        folder=out/name;folder.mkdir();write_json(out/'status.json',dict(status='RUNNING',phase=name))
        wrapper=student if period is None else CachedUpdateParts(student,period=period,tile=a.tile,repair=mode)
        a.stage='full' if a.frames==0 else 'pilot';torch.cuda.reset_peak_memory_stats()
        report=evaluate(a,teacher,cfg,rows,folder,wrapper)
        records=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
        expected=rows if a.frames==0 else rows[:a.frames]
        assert [(f['frame'],f['segment'],f['supervised']) for f in records]==[(f['frame'],f['segment'],f['supervised']) for f in expected]
        report.update(checkpoint_sha256=sha,source_epoch=saved['epoch'],variant=variant,period=period,repair=mode,
            caveats=['Fixed early checkpoint, no sparse training; prefix AP is not full validation AP.',
                     'Cache includes actual previous state, residual/gate logits and age; current projection/readout stay dense.',
                     'Same GPU sequential model timing includes cache/warp/repair/current/readout, other GPUs may be training.',
                     'Policy differs from freezing final state; compare with controls in this run, not older epoch results.'])
        if period is not None:
            report['parts_summary']={k:float(np.mean([f[k] for f in records])) for k in
                ['parts_anchor','parts_fallback','parts_write_fraction','parts_input_support_fraction']}
            assert max(f['max_parts_age'] for f in records)<.250001
        write_json(folder/'report.json',report);reports[name]=report
    assert reports['dense_before']['metrics']==reports['dense_after']['metrics']
    left=[json.loads(s) for s in (out/'masked_period3/frames.jsonl').read_text().splitlines()]
    right=[json.loads(s) for s in (out/'period3/frames.jsonl').read_text().splitlines()]
    assert all(a['parts_write_fraction']==b['parts_write_fraction'] and a['parts_fallback']==b['parts_fallback'] for a,b in zip(left,right))
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint_sha256=sha,reports=reports,
        masked_gather_write_counts_equal=True,masked_gather_ap_equal=reports['masked_period3']['metrics']==reports['period3']['metrics']))
    write_json(out/'status.json',dict(status='COMPLETED'))


if __name__=='__main__':main()
