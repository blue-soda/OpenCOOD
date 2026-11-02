"""Fine-cache frozen execution, period1 equivalence and no-step gradient checks."""
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


def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text())
    assert audit['status']==gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    convs=[m for m in source.core.readout.modules() if isinstance(m,nn.Conv2d)];assert len(convs)==4
    sums=[torch.zeros(m.in_channels,dtype=torch.float64) for m in convs];sizes=[0]*len(convs);handles=[]
    for i,conv in enumerate(convs):
        def collect(module,args,i=i):
            x=args[0];sums[i].add_(x.square().sum((0,2,3),dtype=torch.float64).cpu());sizes[i]+=x.shape[0]*x.shape[2]*x.shape[3]
        handles.append(conv.register_forward_pre_hook(collect))
    _,stream=loader(rows[:512],cfg,2,303);state=base_state=None;source_hash=model_digest(source)
    with torch.no_grad():
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,sample['meta']);_,state,_=source.step_spatial(x,state,sample['meta'])
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='calibration',scans=i+1,total=512))
    for handle in handles:handle.remove()
    assert model_digest(source)==source_hash
    rms=[(value/size).sqrt() for value,size in zip(sums,sizes)];quanta=[(.125*v).clamp_min(1e-4).float().tolist() for v in rms]
    calibration=dict(frames=[r['frame'] for r in rows[:512]],relative_quantum=.125,floor=1e-4,layer_input_rms=[v.tolist() for v in rms],quanta=quanta,elements_per_channel=sizes,selection='Fixed0.125RMS per input channel on train512, no labels/AP; no threshold search')
    write_json(out/'calibration.json',calibration);models=dict(control_p4=source,events_p4=ReadoutEventDetector(source,quanta).cuda().eval(),control_p1=copy.deepcopy(source));models['control_p1'].period=1;models['events_p1']=ReadoutEventDetector(models['control_p1'],quanta).cuda().eval()
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];rows=rows[512:544]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),trained_sha256=gate['training']['protected_light']['checkpoint_sha256'],verification_sha256=digest(Path(a.verification)),initialization_sha256=digest(Path(a.initialization)),calibration_sha256=digest(out/'calibration.json'),frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),runtime=dict(torch_version=torch.__version__,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),boundary='Dense GPU emulation of causal per-layer change encoding; actual Conv MACs not skipped. Sending references fixed tensor coordinates, no warp/Conv commutation. Existing upstream task memory unchanged; added readout execution state is not independent task memory. Conditional unit-AC/weighted-event-MAC needs persistent hardware accumulators, all refresh dense MACs included. No AP/latency/optimizer.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                old=states[n];(pred,states[n],info),cost=measured_step(m,x,old,meta);preds[n]=pred;assert all(torch.isfinite(v).all() for v in pred.values())
                row['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],reset=info['reset'],events=info.get('readout_events'))
                if n.startswith('events') and old is not None and not info['full_refresh']:
                    errors=[];convolutions=[layer for layer in m.core.readout.layers if isinstance(layer,nn.Conv2d)]
                    for conv,previous,current in zip(convolutions,old['readout'],states[n]['readout']):
                        before=conv(previous);after=conv(current);delta=F.conv2d(current-previous,conv.weight,None,conv.stride,conv.padding)
                        assert torch.allclose(before+delta,after,atol=5e-4,rtol=1e-4);errors.append(float((before+delta-after).abs().max()))
                    row['arms'][n]['linear_delta_max_errors']=errors
            for period in (1,4):
                left,right='control_p%d'%period,'events_p%d'%period
                assert all(torch.equal(states[left][k],states[right][k]) for k in base.state_keys)
                assert row['arms'][left]['core_conv_flops']==row['arms'][right]['core_conv_flops']
                if row['arms'][left]['refresh']:assert all(torch.equal(preds[left][k],preds[right][k]) for k in preds[left])
                row['arms'][right]['prediction_nmse']=sum(float((preds[right][k]-preds[left][k]).square().mean()/(preds[left][k].square().mean()+1e-6)) for k in preds[left])/len(preds[left])
            records.append(row);f.write(json.dumps(row)+'\n')
    assert len(records)==32 and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/32,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=32,weights_unchanged=True,paired_task_states_exact=True,refresh_predictions_exact=True,linear_delta_identity_passed=True,calibration_sha256=digest(out/'calibration.json'),trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min calibrated readout event engineering')));signal.alarm(2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
