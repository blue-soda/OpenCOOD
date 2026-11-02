"""Fine-cache frozen execution, period1 equivalence and no-step gradient checks."""
import argparse,copy,json,signal,subprocess,sys
from pathlib import Path
from collections import defaultdict
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
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved;rows=rows[512:544]
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    models=dict(control_p4=source,control_p1=copy.deepcopy(source));models['control_p1'].period=1
    for n,light,period in (('fine_dense_p1',False,1),('fine_light_p1',True,1),('fine_dense_p4',False,4),('fine_light_p4',True,4)):
        torch.manual_seed(303);models[n]=FineRefreshDetector(source,light,period).cuda().eval()
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};assert len({hashes[n] for n in models if n.startswith('fine_')})==1
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),trained_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),verification_sha256=digest(Path(a.verification)),compact_verification_sha256=digest(Path(a.compact_verification)),frames=[r['frame'] for r in rows],seed=303,precision=dict(matmul_tf32=False,cudnn_tf32=False),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        boundary='Frozen32 engineering, no optimizer/AP/latency. New fine light random initialization requires future training. Full-refresh functions preserved with own history; only period1 and reset predictions must equal control. Core Conv2d/Transpose2MAC only, excludes Pillar/warp/BN/activation/nonconv.')
    write_json(out/'contract.json',contract);counts={n:defaultdict(int) for n in models};handles=[];states={n:None for n in models};base_state=None;records=[];first=[]
    for n,m in models.items():
        for name,mod in m.core.named_modules():
            if isinstance(mod,(nn.Conv2d,nn.ConvTranspose2d)):
                def hit(module,args,result,n=n,g=group(name)):
                    mac=args[0].numel()*(module.out_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1] if isinstance(module,nn.ConvTranspose2d) else result.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
                    counts[n][g]+=2*mac
                handles.append(mod.register_forward_hook(hit))
    _,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);entry=dict(frame=meta['frame'],spatial_shape=list(x.shape),arms={});preds={}
            if i<4:first.append((x.detach(),copy.deepcopy(meta)))
            for n,m in models.items():
                counts[n].clear();pred,states[n],info=m.step_spatial(x,states[n],meta);preds[n]=pred;assert all(torch.isfinite(v).all() for v in pred.values())
                entry['arms'][n]=dict(groups=dict(counts[n]),core_conv_flops=sum(counts[n].values()),refresh=info['full_refresh'],reset=info['reset'],cache_bytes=info['cache_bytes'])
                if n.startswith('fine_'):
                    expected_fine=bool(info['full_refresh']) or 'dense' in n
                    assert bool(counts[n]['observation'])==expected_fine
                    assert bool(counts[n]['fine_light'])==(not expected_fine)
            for n in ('fine_dense_p1','fine_light_p1'):
                assert all(torch.equal(preds[n][k],preds['control_p1'][k]) for k in preds[n])
                assert all(torch.equal(states[n][k],states['control_p1'][k]) for k in base.state_keys)
            if entry['arms']['control_p4']['reset']:
                assert all(torch.equal(preds[n][k],preds['control_p4'][k]) for n in ('fine_dense_p4','fine_light_p4') for k in preds[n])
            f.write(json.dumps(entry)+'\n');records.append(entry)
    for h in handles:h.remove()
    gradients={}
    for n in ('fine_dense_p4','fine_light_p4'):
        model=models[n];prefixes=('core.fine_update.',)+(('core.fine_light.',) if n=='fine_light_p4' else ())
        for name,p in model.named_parameters():p.requires_grad_(name.startswith(prefixes))
        state=None
        for x,meta in first:pred,state,info=model.step_spatial(x,state,meta,detach_state=False)
        sum(v.square().mean() for v in pred.values()).backward();norms={}
        for name,p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all();norms[name]=float(p.grad.abs().sum())
            else:assert p.grad is None
        for prefix in prefixes:assert sum(value for name,value in norms.items() if name.startswith(prefix))>0
        assert state['value'].grad_fn is not None;gradients[n]=dict(status='PASSED',parameter_l1_gradients=norms,own_fine_state_has_graph=True,no_optimizer_step=True)
        model.zero_grad(set_to_none=True)
        for p in model.parameters():p.requires_grad_(False)
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    initial=out/'fine_initialization.pth';torch.save(dict(fine_update=models['fine_light_p4'].core.fine_update.state_dict(),fine_light=models['fine_light_p4'].core.fine_light.state_dict(),contract=contract),initial)
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(x['arms'][n]['core_conv_flops'] for x in records)/32,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=None,ap70=None,refresh_frames=sum(bool(x['arms'][n]['refresh']) for x in records)) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,gradients=gradients,scans=32,period1_predictions_states_exact=True,reset_predictions_exact=True,weights_unchanged=True,trained=False,ap_evaluated=False,initialization_sha256=digest(initial)));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min fine cache engineering')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
