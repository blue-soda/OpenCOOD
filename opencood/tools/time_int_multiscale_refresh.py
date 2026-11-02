"""Exclusive-card paired timing of untrained middle/deep light paths, no AP."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector,MultiscaleRefreshDetector
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def exclusive(uuid):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits']).decode().strip().splitlines()
    peers=[int(r.split(',')[0]) for r in rows if r.split(',')[-1].strip()==uuid and int(r.split(',')[0])!=os.getpid()]
    if peers:raise RuntimeError('Timing contaminated by other GPU processes: '+str(peers))


def worker(a):
    out=Path(a.output);uuid=os.environ['CUDA_VISIBLE_DEVICES'];exclusive(uuid)
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.gate_verification).read_text())
    assert audit['status']==gate['status']=='VERIFIED' and gate['run']=='R-125'
    assert audit['checkpoint_sha256']==gate['checkpoint_sha256']==digest(Path(a.checkpoint))
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    torch.cuda.set_per_process_memory_fraction(.5)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);rows=rows[512:768];write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    # Reproduce the R125 construction order, which determines random new weights.
    candidates={n:TemporalPyramidDetector(base,n).cuda().eval() for n in ('original','gate','leaky')}
    candidates['dual_original']=MultiscaleRefreshDetector(base,'original').cuda().eval()
    candidates['dual_gate']=MultiscaleRefreshDetector(base,'gate').cuda().eval()
    models={n:candidates[n] for n in ('original','dual_gate')};del candidates,teacher,saved
    for m in models.values():
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()}
    write_json(out/'contract.json',dict(checkpoint_sha256=digest(Path(a.checkpoint)),gate_verification_sha256=digest(Path(a.gate_verification)),
        uuid=uuid,device=torch.cuda.get_device_name(),torch_version=torch.__version__,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        frames=[r['frame'] for r in rows],passes=2,warmup_scans_per_pass=16,seed=303,period=4,
        initial_model_hashes=hashes,timing='torch.cuda.synchronize before/after model.step, includes pillar/warp/readout/heads, excludes loader/H2D/postprocess; no Conv hooks',
        boundary='Untrained lightweight projections and gates; engineering latency only, no AP or matched-quality speed claim'))
    records=[]
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for repeat in range(2):
            states={n:None for n in models};_,stream=loader(rows,cfg,2,303)
            for i,sample in enumerate(stream):
                if i%32==0:exclusive(uuid)
                meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'))
                order=['original','dual_gate'] if (i+repeat)%2==0 else ['dual_gate','original']
                entry=dict(repeat=repeat,index=i,frame=meta['frame'],included=i>=16,arms={})
                for n in order:
                    torch.cuda.synchronize();start=time.perf_counter()
                    pred,states[n],info=models[n].step(ego,states[n],meta)
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                    entry['arms'][n]=dict(model_step_ms=elapsed,full_refresh=info.get('full_refresh'),reset=info['reset'])
                log.write(json.dumps(entry)+'\n');records.append(entry)
                if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='timing',repeat=repeat,scans=i+1,total=256))
    exclusive(uuid);assert all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={}
    for n,m in models.items():
        values=[r['arms'][n]['model_step_ms'] for r in records if r['included']]
        assert len(values)==480
        metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),model_step_ms_mean=float(np.mean(values)),
            model_step_ms_p50=float(np.median(values)),model_step_ms_p95=float(np.percentile(values,95)),samples=480,
            per_pass_ms=[float(np.mean([r['arms'][n]['model_step_ms'] for r in records if r['included'] and r['repeat']==rep])) for rep in range(2)],
            core_conv_flops_mean=None,ap50=None,ap70=None,scope='No hooks during timing; separate R125 core budget is real32, not this256 timing sample')
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,timing_valid=True,trained=False,ap_evaluated=False,
        relative_change=metrics['dual_gate']['model_step_ms_mean']/metrics['original']['model_step_ms_mean']-1,
        boundary='Same-card counterbalanced engineering timing; unknown candidate AP, not matched-quality acceleration'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','gate-verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='identity'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min independent timing budget')));signal.alarm(2700)
    try:worker(a)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False));raise


if __name__=='__main__':main()
