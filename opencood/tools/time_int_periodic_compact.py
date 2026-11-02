"""StrictFP32 paired P4/P5/P6 original and rank128 readout timing."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import sys
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector,ProtectedMultiscaleRefreshDetector
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.models.sub_modules.int_factorized_readout import compact_readout



def exclusive(uuid):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits']).decode().strip().splitlines()
    peers=[int(r.split(',')[0]) for r in rows if r.split(',')[-1].strip()==uuid and int(r.split(',')[0])!=os.getpid()]
    if peers:raise RuntimeError('Timing contaminated by other GPU processes: '+str(peers))


def worker(a):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);uuid=os.environ['CUDA_VISIBLE_DEVICES'];exclusive(uuid)
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.gate_verification).read_text())
    assert audit['status']==gate['status']=='VERIFIED' and gate['run']=='R-139'
    assert gate['checkpoint_sha256']==digest(Path(a.checkpoint))
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    torch.cuda.set_per_process_memory_fraction(.5)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);rows=rows[512:768];write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    compact_gate=json.loads(Path(a.compact_verification).read_text());assert compact_gate['status']=='VERIFIED' and compact_gate['run']=='R-145'
    assert digest(Path(a.initialization))==compact_gate['rank128_checkpoint_sha256']
    models={};initial=torch.load(a.initialization,map_location='cpu')['readout']
    for period in (4,5,6):
        for compact in (False,True):
            n='p%d_%s'%(period,'rank128' if compact else 'dense');m=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(m,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256']);m.period=period;m.max_age=period*.1+.05
            if compact:m.core.readout,_=compact_readout(base.core.readout,128);m.core.readout.load_state_dict(initial);m.eval()
            models[n]=m
    del teacher,saved
    for m in models.values():
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()}
    write_json(out/'contract.json',dict(checkpoint_sha256=digest(Path(a.checkpoint)),gate_verification_sha256=digest(Path(a.gate_verification)),
        uuid=uuid,device=torch.cuda.get_device_name(),torch_version=torch.__version__,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        frames=[r['frame'] for r in rows],passes=2,warmup_scans_per_pass=16,seed=303,source_trained_sha256=dict(protected_light=gate['training']['protected_light']['checkpoint_sha256']),initialization_sha256=digest(Path(a.initialization)),precision=dict(matmul_tf32=False,cudnn_tf32=False),
        initial_model_hashes=hashes,timing='torch.cuda.synchronize before/after model.step, includes pillar/warp/readout/heads, excludes loader/H2D/postprocess; no Conv hooks',
        boundary='Fixed R139 P4-trained cache plus R145 rank128, six P4/P5/P6 arms. Longer-period rank128 quality unmeasured; timing cannot imply acceptable AP. GPU exclusive but node shared. StrictFP32, train256 timing only.'))
    records=[]
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for repeat in range(2):
            states={n:None for n in models};_,stream=loader(rows,cfg,2,303)
            for i,sample in enumerate(stream):
                if i%32==0:exclusive(uuid)
                meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'))
                names=list(models);offset=i%len(names);order=names[offset:]+names[:offset]
                if repeat%2:order=list(reversed(order))
                entry=dict(repeat=repeat,index=i,frame=meta['frame'],included=i>=16,execution_order=order,arms={})
                for n in order:
                    torch.cuda.synchronize();start=time.perf_counter()
                    pred,states[n],info=models[n].step(ego,states[n],meta)
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                    cost=264923366400 if info['full_refresh'] else 139535827200
                    if 'rank128' in n:cost-=33855897600
                    entry['arms'][n]=dict(model_step_ms=elapsed,full_refresh=info['full_refresh'],
                        reset=info['reset'],core_conv_flops_reconstructed=cost)
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
            core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops_reconstructed'] for r in records if r['included'])/480,
            ap50=None,ap70=None,scope='No hooks during timing; core FLOPs reconstructed from actual refresh/patch schedule and independently validated constant-grid module costs')
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,timing_valid=True,trained=False,ap_evaluated=False,
        relative_to='p4_rank128',
        relative_changes={n:m['model_step_ms_mean']/metrics['p4_rank128']['model_step_ms_mean']-1 for n,m in metrics.items()},
        boundary='Trained weights fixed, same-card counterbalanced train256 timing; AP separately evaluated, quality loss must be retained in any acceleration claim.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','gate-verification','trained','compact-verification','initialization','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='identity'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min independent timing budget')));signal.alarm(2700)
    try:worker(a)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False));raise


if __name__=='__main__':main()
