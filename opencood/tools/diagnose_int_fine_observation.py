"""Frozen fine-observation and own-state mismatch diagnostics; no quality search."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_fine_refresh import FineRefreshDetector
from opencood.models.sub_modules.int_factorized_readout import compact_readout
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def worker(a,out):
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text());fv=json.loads(Path(a.fine_verification).read_text())
    assert audit['status']==gate['status']==cv['status']==fv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145' and fv['run']=='R-158'
    assert audit['frames']==768
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256']==fv['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256'] and digest(Path(a.fine_initialization))==fv['initialization_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval();models=dict(frozen=source)
    fine=torch.load(a.fine_initialization,map_location='cpu')
    for n,light in (('initial_light',True),('final_light',True),('final_dense',False)):
        torch.manual_seed(303);m=FineRefreshDetector(source,light).cuda().eval();m.core.fine_update.load_state_dict(fine['fine_update']);m.core.fine_light.load_state_dict(fine['fine_light']);models[n]=m
    lv=json.loads(Path(a.learning_verification).read_text());assert lv['status']=='VERIFIED' and lv['run']=='R-162'
    for n,source_name in (('final_light','fine_light'),('final_dense','fine_dense')):load_targets(models[n],Path(a.learned)/(source_name+'_final.pth'),lv['training'][source_name]['checkpoint_sha256'])
    for m in [base]+list(models.values()):
        m.eval()
        for parameter in m.parameters():parameter.requires_grad_(False)
    rows=rows[512:544];states={n:None for n in models};base_state=None;hashes={n:model_digest(m) for n,m in models.items()};captured={};handles=[];records=[]
    for n,m in models.items():
        for name in ('observation','fine_light'):
            if hasattr(m.core,name):handles.append(getattr(m.core,name).register_forward_hook(lambda mod,inp,value,n=n:captured.update({n:value})))
    contract=dict(frames=[r['frame'] for r in rows],checkpoint_sha256=digest(Path(a.checkpoint)),learning_verification_sha256=digest(Path(a.learning_verification)),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),boundary='Frozen32 train diagnostic. Hooks record fine current observation and own states only; no optimizer/AP/timing. Comparison can locate mismatch, not establish capacity or convergence causally.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={})
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);assert all(torch.isfinite(v).all() for v in pred.values())
                cur=captured[n];target=captured['frozen'];state=states[n]['value'];gold=states['frozen']['value']
                row['arms'][n]=dict(refresh=info['full_refresh'],core_conv_flops=cost,observation_nmse=float((cur-target).square().mean()/(target.square().mean()+1e-6)),observation_rms_ratio=float((cur.square().mean()/(target.square().mean()+1e-6)).sqrt()),observation_negative_fraction=float((cur<0).float().mean()),fine_state_nmse=float((state-gold).square().mean()/(gold.square().mean()+1e-6)))
            records.append(row);f.write(json.dumps(row)+'\n')
    for handle in handles:handle.remove()
    assert len(records)==32 and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={}
    for n,m in models.items():
        ordinary=[r['arms'][n] for r in records if not r['arms'][n]['refresh']]
        metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/32,model_step_ms_mean=None,ap50=None,ap70=None,ordinary_frames=len(ordinary),ordinary_means={k:sum(r[k] for r in ordinary)/len(ordinary) for k in ('observation_nmse','observation_rms_ratio','observation_negative_fraction','fine_state_nmse')})
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,weights_unchanged=True,trained=False,ap_evaluated=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','fine-initialization','learning-verification','learned','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min frozen fine-cache diagnostics')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
