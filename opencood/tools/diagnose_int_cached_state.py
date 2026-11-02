"""Frozen failure localization, no AP and no extra learning or search."""
import argparse,copy,json,signal,subprocess
from pathlib import Path
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import MultiscaleRefreshDetector
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def load_targets(model,path,expected):
    assert digest(path)==expected
    saved=torch.load(path,map_location='cpu');params=dict(model.named_parameters())
    with torch.no_grad():
        for n,value in saved['targets'].items():params[n].copy_(value)
    return saved


def worker(a,out):
    verified=json.loads(Path(a.verification).read_text());assert verified['status']=='VERIFIED' and verified['run']=='R-130'
    audit=json.loads(Path(a.audit).read_text());assert audit['status']=='VERIFIED' and audit['frames']==768
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    assert digest(Path(a.checkpoint))==verified['checkpoint_sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model']);del teacher,saved
    models={}
    for n,period in (('dense_adapted',1),('light_own',4),('light_dense',1),('light_teacher_history',4)):
        torch.manual_seed(303);m=MultiscaleRefreshDetector(base,'gate',period).cuda().eval()
        source='dense_gate' if n=='dense_adapted' else 'light_p4'
        load_targets(m,Path(a.trained)/(source+'_final.pth'),verified['training'][source]['checkpoint_sha256']);models[n]=m
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};base_hash=model_digest(base)
    rows=rows[512:768];contract=dict(frames=[r['frame'] for r in rows],verification_sha256=digest(Path(a.verification)),
        checkpoint_sha256=digest(Path(a.checkpoint)),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        boundary='Frozen feature localization on train256, no AP/training/timing; light_teacher_history borrows prior light_dense features and is an oracle diagnostic, never deployable.')
    write_json(out/'contract.json',contract);states={n:None for n in models};base_state=None;records=[]
    _,stream=loader(rows,cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));outputs={}
            handle=base.core.readout.register_forward_hook(lambda m,x,y:outputs.update(parent=y))
            try:x,base_state=capture(base,ego,base_state,meta)
            finally:handle.remove()
            previous_dense=states['light_dense'];entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                state=states[n]
                if n=='light_teacher_history' and state is not None:
                    state=dict(state)
                    for key in m.state_keys:state[key]=previous_dense[key]
                handle=m.core.readout.register_forward_hook(lambda mod,args,y:outputs.update(current=y))
                try:(pred,states[n],info),cost=measured_step(m,x,state,meta)
                finally:handle.remove()
                errors={k:float((states[n][k]-base_state[k]).square().mean()/(base_state[k].square().mean()+1e-6)) for k in m.state_keys}
                errors['readout']=float((outputs['current']-outputs['parent']).square().mean()/(outputs['parent'].square().mean()+1e-6))
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                entry['arms'][n]=dict(normalized_mse=errors,full_refresh=info['full_refresh'],age=states[n]['since_refresh'],
                    core_conv_flops=cost,cache_bytes=info['cache_bytes'])
            log.write(json.dumps(entry)+'\n');records.append(entry)
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=i+1,total=256))
    assert model_digest(base)==base_hash and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={}
    for n,m in models.items():
        entries=[r['arms'][n] for r in records];groups={}
        for age in range(4):
            group=[x for x in entries if x['age']==age]
            if group:groups[str(age)]=dict(frames=len(group),normalized_mse={k:sum(x['normalized_mse'][k] for x in group)/len(group) for k in ('value','coarse','deep','readout')})
        metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(x['core_conv_flops'] for x in entries)/256,
            model_step_ms_mean=None,ap50=None,ap70=None,cache_bytes=entries[0]['cache_bytes'],by_age=groups)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=256,weights_unchanged=True,trained=False,ap_evaluated=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min frozen localization')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
