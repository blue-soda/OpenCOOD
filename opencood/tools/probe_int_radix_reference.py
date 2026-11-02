"""Bounded channel-event Conv backend proof on fixed causal readout streams."""
import argparse,copy,json,signal,subprocess,sys
from pathlib import Path
from collections import defaultdict
from opencood.models.sub_modules.int_readout_events import ReadoutEventDetector
from opencood.tools.check_int_pyramid_cache import measured_step
from torch.nn import functional as F
import torch
from torch import nn
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_fine_refresh import FineRefreshDetector
from opencood.models.sub_modules.int_factorized_readout import compact_readout
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.profile_int_compact_cache import group
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest

from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadoutDetector
from opencood.models.sub_modules.int_radix_integer_reference import radix_integer_reference
from opencood.models.sub_modules.int_integer_event_reference import dense_integer_reference
from opencood.tools.time_int_fine_rank64 import exclusive
import os,time
import numpy as np
from opencood.utils import eval_utils
from opencood.tools.int_artifact_paths import artifact_path


def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text())
    assert (audit.get('status')=='VERIFIED' if not a.full else audit['frames']==5794 and len(audit['files'])==7532);assert gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for r in audit['files']:
        path=Path(artifact_path(r['path'])) if a.full else Path(r['relocated_path']);assert digest(path)==r['sha256']
    a.mode='gru';a.stage='full' if a.full else 'calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    ev=json.loads(Path(a.event_verification).read_text());rv=json.loads(Path(a.reference_verification).read_text());assert ev['status']==rv['status']=='VERIFIED' and ev['run']=='R-172' and rv['run']==('R-148' if a.full else 'R-168')
    event_contract=Path(a.calibration).parent/'contract.json';assert digest(event_contract)==ev['source_sha256']['contract.json'];ec=json.loads(event_contract.read_text());assert ec['verification_sha256']==digest(Path(a.verification)) and ec['initialization_sha256']==digest(Path(a.initialization)) and ev['checkpoint_sha256']==gate['checkpoint_sha256']
    assert ev['calibration_sha256']==digest(Path(a.calibration));calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    iv=json.loads(Path(a.integer_verification).read_text());assert iv['status']=='VERIFIED' and iv['run']=='R-181'
    icp=Path(a.integer_source)/'contract.json';assert digest(icp)==iv['source_sha256']['contract.json'];ic=json.loads(icp.read_text());assert ic['checkpoint_sha256']==gate['checkpoint_sha256'] and ic['calibration_sha256']==ev['calibration_sha256']
    rows=rows[512:516];model=IntegerReadoutDetector(source,calibration['quanta'],False).cuda().eval()
    for m in (base,model):
        for parameter in m.parameters():parameter.requires_grad_(False)
    uuid=os.environ['CUDA_VISIBLE_DEVICES'];exclusive(uuid);before=model_digest(model);state=base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),integer_verification_sha256=digest(Path(a.integer_verification)),calibration_sha256=digest(Path(a.calibration)),frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),uuid=uuid,device=torch.cuda.get_device_name(),torch_version=torch.__version__,precision=dict(matmul_tf32=False,cudnn_tf32=False),warmup_per_layer_backend=1,repeats=2,partial_chunk=256,boundary='Reference tooling only:16 real integer Conv inputs, FP64 reference vs signed radix256 FP32 GEMMs each at most256 products, absolute partial bound256*255^2<2^24; final int64 shift/add within original absolute bound<2^53. Bit-exact output required. Paired synchronized complete backend-function microtiming includes unfold/digit decomposition/casts/range checks, excludes source model/frontend/BN/head. GPU exclusive/node shared; not model latency, native integer kernel, SNN acceleration or AP.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,sample['meta']);_,state,_=model.step_spatial(x,state,sample['meta']);row=dict(frame=sample['meta']['frame'],layers=[])
            convs=[layer for layer in model.core.readout.layers if isinstance(layer,nn.Conv2d)]
            for i,(conv,inputs,gold) in enumerate(zip(convs,state['readout'],state['readout_accumulators'])):
                weight=getattr(model.core.readout,'integer_weight_%d'%i);calls={'fp64':lambda:dense_integer_reference(conv,inputs,weight),'radix':lambda:radix_integer_reference(conv,inputs,weight)}
                for fn in calls.values():result,_=fn();assert torch.equal(result,gold)
                timings=[];stats=None
                for repeat in range(2):
                    order=['fp64','radix'] if repeat==0 else ['radix','fp64'];timed=dict(repeat=repeat,order=order)
                    for name in order:
                        exclusive(uuid);torch.cuda.synchronize();start=time.perf_counter();result,cost=calls[name]();torch.cuda.synchronize();timed[name+'_ms']=(time.perf_counter()-start)*1000;assert torch.equal(result,gold)
                        if name=='radix':stats=cost
                    timings.append(timed)
                row['layers'].append(dict(layer=i,input_channels=conv.in_channels,output_channels=conv.out_channels,kernel=conv.kernel_size[0],height=inputs.shape[-2],width=inputs.shape[-1],bit_exact=True,fp64_reference_flops=2*inputs.shape[-2]*inputs.shape[-1]*conv.in_channels*conv.out_channels*conv.kernel_size[0]**2,radix=stats,timings=timings))
            records.append(row);f.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=4))
    assert len(records)==4 and model_digest(model)==before;exclusive(uuid);layers=[v for row in records for v in row['layers']];times=[v for layer in layers for v in layer['timings']]
    metrics=dict(parameters=sum(p.numel() for p in model.parameters()),core_conv_flops_mean=None,fp64_reference_flops_total=sum(v['fp64_reference_flops'] for v in layers),radix_fp32_reference_flops_total=sum(v['radix']['fp32_reference_flops'] for v in layers),model_step_ms_mean=None,ap50=None,ap70=None)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=4,linear_checks=16,weights_unchanged=True,bit_exact=True,microtiming=dict(samples_per_backend=len(times),mean_ms={n:sum(t[n+'_ms'] for t in times)/len(times) for n in ('fp64','radix')}),trained=False,ap_evaluated=False,timing_valid=False,backend_microtiming_valid=True));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','event-verification','calibration','reference-verification','integer-verification','integer-source','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.full=False;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min bounded radix integer reference proof')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
