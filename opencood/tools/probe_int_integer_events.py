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

from opencood.models.sub_modules.int_integer_event_reference import folded_integer_weights,integer_input,dense_integer_reference,sparse_integer_delta
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
    rows=rows[512:516];model=ReadoutEventDetector(source,calibration['quanta']).cuda().eval()
    for m in (base,model):
        for parameter in m.parameters():parameter.requires_grad_(False)
    before_hash=model_digest(model);state=base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),event_verification_sha256=digest(Path(a.event_verification)),calibration_sha256=digest(Path(a.calibration)),initialization_sha256=digest(Path(a.initialization)),frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),integer_weights_bits=16,integer_accumulator_bits=64,reference_dtype='float64 with proven absolute-sum bound below2**53',chunk_events=32768,boundary='Four fixed real frames, no AP/timing. R172 causal q streams are projected onto fixed integer grids; folded W*quantum quantized int16 with per-output scales. Changes refresh model; not original ANN equivalence. Actual sparse int64 MAC update must be bit-exact to bounded FP64 dense integer reference. Diagnostic reference excluded from backend cost. No integrated nonlinear detector/hardware claim.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);old=state;_,state,info=model.step_spatial(x,state,meta)
            row=dict(frame=meta['frame'],refresh=info['full_refresh'],layers=[])
            if not info['full_refresh']:
                convs=[m for m in model.core.readout.layers if isinstance(m,nn.Conv2d)]
                for i,(conv,previous,current) in enumerate(zip(convs,old['readout'],state['readout'])):
                    quantum=getattr(model.core.readout,'quantum_%d'%i);weight,scale=folded_integer_weights(conv,quantum);old_int=integer_input(previous,quantum);new_int=integer_input(current,quantum)
                    before,bound_before=dense_integer_reference(conv,old_int,weight);after,bound_after=dense_integer_reference(conv,new_int,weight)
                    calls=[];handle=conv.register_forward_hook(lambda *_:calls.append(1));sparse,stats=sparse_integer_delta(conv,new_int-old_int,weight);handle.remove();assert not calls
                    equal=torch.equal(before+sparse,after);stats.update(layer=i,integer_bit_exact=equal,dense_conv_calls_inside_backend=len(calls),reference_absolute_bound=max(bound_before,bound_after),quantized_weight_bytes=weight.numel()*weight.element_size(),scale_bytes=scale.numel()*scale.element_size(),max_input_integer=int(new_int.abs().max()))
                    row['layers'].append(stats);write_json(out/'last_layer.json',dict(frame=meta['frame'],**stats));assert equal
            records.append(row);f.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=4))
    assert len(records)==4 and sum(len(row['layers']) for row in records)==8 and model_digest(model)==before_hash
    layers=[v for row in records for v in row['layers']];metrics=dict(parameters=sum(p.numel() for p in model.parameters()),core_conv_flops_mean=None,integer_backend_macs_total=sum(v['executed_macs'] for v in layers),integer_backend_operations_total=2*sum(v['executed_macs'] for v in layers),dense_reference_flops_total=2*sum(v['dense_macs'] for v in layers),model_step_ms_mean=None,ap50=None,ap70=None)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=4,ordinary_frames=2,linear_checks=8,weights_unchanged=True,no_dense_conv_inside_backend=True,integer_bit_exact=True,trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','event-verification','calibration','reference-verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.full=False;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min four-frame integer event reference proof')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
