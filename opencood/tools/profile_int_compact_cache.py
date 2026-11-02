"""Executed named-core module breakdown of compact/cache composition, no AP."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
from collections import defaultdict
import torch
from torch import nn
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector,ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_factorized_readout import compact_readout
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def group(name):
    parts=name.split('.')
    return '.'.join(parts[:2]) if parts[0] in ('deblocks','refresh_updates') else parts[0]


def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text());assert audit['status']==gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved;rows=rows[512:544]
    models={n:TemporalPyramidDetector(base,'original').cuda().eval() for n in ('frozen','initial_rank128')}
    for n in ('protected','protected_initial_rank128'):
        m=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(m,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256']);models[n]=m
    initial=torch.load(a.initialization,map_location='cpu')['readout']
    for n in ('initial_rank128','protected_initial_rank128'):models[n].core.readout,_=compact_readout(base.core.readout,128);models[n].core.readout.load_state_dict(initial);models[n].eval()
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];handles=[];counts={};groups={}
    for n,m in models.items():
        counts[n]=defaultdict(int);groups[n]=defaultdict(int)
        for name,p in m.core.named_parameters():groups[n][group(name)]+=p.numel()
        for name,module in m.core.named_modules():
            if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d)):
                def count(mod,args,result,n=n,g=group(name)):
                    mac=args[0].numel()*mod.out_channels//mod.groups*mod.kernel_size[0]*mod.kernel_size[1] if isinstance(mod,nn.ConvTranspose2d) else result.numel()*mod.in_channels//mod.groups*mod.kernel_size[0]*mod.kernel_size[1]
                    counts[n][g]+=2*mac
                handles.append(module.register_forward_hook(count))
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),trained_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),frames=[r['frame'] for r in rows],precision=dict(matmul_tf32=False,cudnn_tf32=False),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        scope='Executed core Conv2d/ConvTranspose2d only,2MAC; Pillar VFE/scatter, warps, activations/normalization/reductions/indexing excluded. No instrumented latency claim. Parameters include all core modules, even inactive protected/light alternatives.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);entry=dict(frame=meta['frame'],arms={})
            for n,m in models.items():
                counts[n].clear();pred,states[n],info=m.step_spatial(x,states[n],meta);assert all(torch.isfinite(v).all() for v in pred.values());entry['arms'][n]=dict(groups=dict(counts[n]),total_core_flops=sum(counts[n].values()),refresh=info.get('full_refresh','dense'),cache_bytes=info['cache_bytes'])
            records.append(entry);f.write(json.dumps(entry)+'\n')
    for h in handles:h.remove()
    assert all(model_digest(m)==hashes[n] for n,m in models.items());metrics={}
    for n,m in models.items():
        phases={p:[r['arms'][n] for r in records if p=='all' or bool(r['arms']['protected']['refresh'])==(p=='refresh')] for p in ('all','refresh','ordinary')}
        breakdown={p:{g:sum(x['groups'].get(g,0) for x in es)/len(es) for g in groups[n]} for p,es in phases.items()}
        metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),core_parameters=sum(groups[n].values()),module_parameters=dict(groups[n]),core_conv_flops_mean=sum(x['total_core_flops'] for x in phases['all'])/32,module_flops_mean=breakdown,phase_counts={p:len(es) for p,es in phases.items()},cache_bytes=phases['all'][0]['cache_bytes'],model_step_ms_mean=None,ap50=None,ap70=None)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=32,weights_unchanged=True,trained=False,ap_evaluated=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min core profile')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
