"""Full all-arm validation of initial and learned subthreshold task states."""
from opencood.tools.pilot_int_causal_readout_qat import *
from opencood.tools.int_artifact_paths import artifact_path
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadoutDetector
from opencood.models.sub_modules.int_compact_causal_integer_state import CompactIntegerReadoutDetector
from torch import nn
from opencood.models.sub_modules.int_subthreshold_membrane import SubthresholdMembraneDetector,quantize_ste

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
    lv=json.loads(Path(a.learning_verification).read_text());assert lv['status']=='VERIFIED' and lv['run']=='R-192';lp=Path(a.learning_source);assert digest(lp/'contract.json')==lv['source_sha256']['contract.json'];lc=json.loads((lp/'contract.json').read_text());assert lc['checkpoint_sha256']==gate['checkpoint_sha256'] and lc['initialization_sha256']==cv['rank128_checkpoint_sha256'] and lc['quantum']==.25
    rv=json.loads(Path(a.reference_verification).read_text());assert rv['status']=='VERIFIED' and rv['run']=='R-184';rp=Path(a.reference_source)/'ap_records.json';assert digest(rp)==rv['source_sha256']['ap_records.json'];gold=json.loads(rp.read_text())['control']
    models=dict(frozen=source);models.update({n+'_initial':SubthresholdMembraneDetector(source,n,.25).cuda().eval() for n in ('carry','drop','reset')});checkpoint_hashes={}
    for n in ('original','carry','drop','reset'):
        path=lp/(n+'_coarse_final.pth');assert digest(path)==lv['training'][n]['checkpoint_sha256'];saved=torch.load(path,map_location='cpu');assert saved['contract']==lc and saved['mode']==n and saved['updates']==lc['expected_update_frames']
        model=SubthresholdMembraneDetector(source,'continuous' if n=='original' else n,.25).cuda().eval();model.core.coarse_update.load_state_dict(saved['coarse']);models[n+'_trained']=model;checkpoint_hashes[n]=digest(path)
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    assert len(rows)==5794 and sum(r['supervised'] for r in rows)==1738
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),learning_verification_sha256=digest(Path(a.learning_verification)),reference_verification_sha256=digest(Path(a.reference_verification)),trained_checkpoint_sha256=checkpoint_hashes,initialization_sha256=digest(Path(a.initialization)),quantum=.25,frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),precision=dict(matmul_tf32=False,cudnn_tf32=False),boundary='Full5794/1738 validation, all four fixed67-update terminal arms plus all distinct initial controls. No best-arm/epoch selection, same parameters/state/core cost, dense hybrid updates remain. Subthreshold carry versus discard changes retained coarse information only; coarse reset retains fine/deep history. No sparse neural hardware or model latency claim; validation-selected, not independent test.')
    write_json(out/'contract.json',contract);dataset,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);assert all(torch.isfinite(z).all() for z in pred.values())
                row['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],membrane=info.get('membrane'))
                if meta['supervised']:
                    batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch);boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            assert all(row['arms'][n]['core_conv_flops']==row['arms']['frozen']['core_conv_flops'] and row['arms'][n]['cache_bytes']==22579200 for n in models)
            records.append(row);f.write(json.dumps(row)+'\n')
            if len(records)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=5794))
    assert len(records)==5794 and all(model_digest(m)==hashes[n] for n,m in models.items());write_json(out/'ap_records.json',ap);saved=json.loads((out/'ap_records.json').read_text());assert saved['frozen']==gold
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,orders={n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in ap.items()}))
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/5794,cache_bytes=22579200,model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=5794,labels=1738,weights_unchanged=True,frozen_raw_ap_exact=True,trained=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))

def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','learning-verification','learning-source','reference-verification','reference-source','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.full=True;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min all-arm subthreshold full validation')));signal.alarm(7200)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
