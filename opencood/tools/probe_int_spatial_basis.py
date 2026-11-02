"""Per-input spatial basis readout engineering; fixed rank4 with full-rank control."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector
from opencood.models.sub_modules.int_spatial_basis_readout import spatial_basis_readout
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def worker(a,out):
    original_precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());assert audit['status']=='VERIFIED'
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved;rows=rows[512:544]
    models={n:TemporalPyramidDetector(base,'original').cuda().eval() for n in ('dense','full_rank','basis4')};changes={}
    for n,rank in (('full_rank',None),('basis4',4)):models[n].core.readout,changes[n]=spatial_basis_readout(base.core.readout,rank);models[n].eval()
    for m in models.values():
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];captures={}
    handles=[m.core.readout.register_forward_hook(lambda mod,args,result,n=n:captures.update({n:result,n+'_input':args[0]})) for n,m in models.items()]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),frames=[r['frame'] for r in rows],basis4_fixed=True,factorizations=changes,svd_initialization='CPU float64 per-input spatial SVD; depthwise basis plus pointwise mixing; cast to float32',original_precision=original_precision,actual_precision=dict(matmul_tf32=False,cudnn_tf32=False),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        boundary='Every frame current-feature compact readout, no extra temporal state. SVD only initializes from parent; full-rank9 is equivalence control with MORE parameters/cost. Basis4 is untrained approximation, no AP or valid timing, not combined with protected cache yet.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);preds={};entry=dict(frame=meta['frame'],arms={})
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);preds[n]=pred
                assert all(torch.isfinite(v).all() for v in pred.values());entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'])
            for n in ('full_rank','basis4'):
                assert all(torch.equal(states[n][k],states['dense'][k]) for k in base.state_keys)
                error=captures[n]-captures['dense'];entry['arms'][n].update(readout_nmse=float(error.square().mean()/(captures['dense'].square().mean()+1e-6)),prediction_max_error=max(float((preds[n][k]-preds['dense'][k]).abs().max()) for k in preds[n]))
            records.append(entry);log.write(json.dumps(entry)+'\n')
            if i==0:
                joined=captures['dense_input'];diagnostic={}
                for label,flags in (('source_defaults',original_precision),('strict_fp32',dict(matmul_tf32=False,cudnn_tf32=False))):
                    torch.backends.cuda.matmul.allow_tf32=flags['matmul_tf32'];torch.backends.cudnn.allow_tf32=flags['cudnn_tf32']
                    dense=models['dense'].core.readout(joined);full=models['full_rank'].core.readout(joined);difference=full-dense
                    diagnostic[label]=dict(flags=flags,readout_max_error=float(difference.abs().max()),readout_nmse=float(difference.square().mean()/(dense.square().mean()+1e-6)))
                write_json(out/'precision_diagnostic.json',diagnostic)
            assert all(torch.allclose(preds['full_rank'][k],preds['dense'][k],atol=5e-4,rtol=1e-4) for k in preds['dense'])
    for h in handles:h.remove()
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    path=out/'basis4_readout.pth';torch.save(dict(readout=models['basis4'].core.readout.state_dict(),contract=contract),path)
    metrics={n:dict(hypothetical_expanded_fp32_state_bytes=(384+256)*4*25200*4 if n=='basis4' else None,parameters=sum(p.numel() for p in m.parameters()),readout_parameters=sum(p.numel() for p in m.core.readout.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/32,readout_nmse_mean=sum(r['arms'][n].get('readout_nmse',0) for r in records)/32,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=32,upstream_states_exact=True,weights_unchanged=True,basis4_checkpoint_sha256=digest(path),trained=False,ap_evaluated=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min spatial basis engineering')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
