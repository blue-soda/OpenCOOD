"""Fresh-process restore of complete frozen fine/readout execution state."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_fine_readout_integer import FineReadoutIntegerDetector
import hashlib,os,time,numpy as np


def tree_record(value):
    if isinstance(value,torch.Tensor):
        a=value.detach().cpu().contiguous().numpy()
        return dict(kind='tensor',dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest(),bytes=value.numel()*value.element_size())
    if isinstance(value,np.ndarray):return dict(kind='ndarray',dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):tree_record(v) for k,v in sorted(value.items())}
    if isinstance(value,(list,tuple)):return dict(kind=type(value).__name__,items=[tree_record(v) for v in value])
    if isinstance(value,np.generic):return value.item()
    return value


def tree_digest(value):
    return hashlib.sha256(json.dumps(tree_record(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def phase_worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.phase=='reference':
        tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text())
    assert (audit.get('status')=='VERIFIED' if not a.validation_prefix else audit['frames']==5794 and len(audit['files'])==7532);assert gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for item in audit['files']:assert digest(Path(artifact_path(item['path'])) if a.validation_prefix else Path(item['relocated_path']))==item['sha256']
    a.mode='gru';a.stage='full' if a.validation_prefix else 'calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    for p in source.parameters():p.requires_grad_(False)
    
    fv=json.loads(Path(a.fine_verification).read_text());assert fv['status']=='VERIFIED' and fv['run']=='R-200'
    assert digest(Path(a.calibration))==fv['source_sha256']['calibration.json'] and fv['checkpoint_sha256']==gate['checkpoint_sha256']
    pv=json.loads(Path(a.primitive_verification).read_text());assert pv['status']=='VERIFIED' and pv['run']=='R-204' and pv['tests']==446
    calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    proof=json.loads(Path(a.fine_stream_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-205' and proof['calibration_sha256']==digest(Path(a.calibration))
    iv=json.loads(Path(a.implicit_verification).read_text());assert iv['status']=='VERIFIED' and iv['run']=='R-201' and iv['checkpoint_sha256']==gate['checkpoint_sha256'] and iv['calibration_sha256']==digest(Path(a.readout_calibration))
    rc=json.loads(Path(a.readout_calibration).read_text());assert rc['relative_quantum']==.125 and rc['floor']==1e-4
    joint=json.loads(Path(a.joint_verification).read_text());assert joint['status']=='VERIFIED' and joint['run']=='R-209' and joint['checkpoint_sha256']==gate['checkpoint_sha256'] and joint['calibration_sha256']==digest(Path(a.calibration)) and joint['readout_calibration_sha256']==digest(Path(a.readout_calibration))
    goldpath=Path(a.joint_source)/'frames.jsonl';assert digest(goldpath)==joint['source_sha256']['frames.jsonl'];gold=[json.loads(s) for s in goldpath.read_text().splitlines()]
    model=FineReadoutIntegerDetector(source,calibration['layers'],rc['quanta'],'hybrid',False).cuda().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    selected=rows[512:544];assert len(selected)==len(gold)==32
    identity=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),fine_calibration_sha256=digest(Path(a.calibration)),readout_calibration_sha256=digest(Path(a.readout_calibration)),joint_verification_sha256=digest(Path(a.joint_verification)),frames=[r['frame'] for r in selected],model_digest=model_digest(model),torch_version=torch.__version__,variant=model.variant,precision=dict(matmul_tf32=False,cudnn_tf32=False))
    root=Path(a.output);write_json(out/'contract.json',dict(identity=identity,phase=a.phase,pid=os.getpid(),uuid=os.environ['CUDA_VISIBLE_DEVICES'],boundary='Frozen inference recovery only; no optimizer/training/AP/timing. Save complete returned state at16, terminate reference process after32, then fresh process resumes final16 from disk and compares prediction/state/statistics hashes. Deterministic per-frame input shuffle, source identities and model digest verified; derived fanout tables rebuilt.'))
    if a.phase=='reference':state=None;start=0
    else:
        metadata=json.loads((root/'checkpoint_metadata.json').read_text());assert metadata['identity']==identity and metadata['next_index']==16 and metadata['checkpoint_sha256']==digest(root/'state_at_016.pth')
        saved=torch.load(root/'state_at_016.pth',map_location='cuda');assert saved['identity']==identity and saved['next_index']==16;state=saved['state'];start=16
        assert tree_digest(state)==metadata['state_hash'] and all(leaf.fanout is None for leaf in model.core.observation.leaves().values())
    records=[];_,stream=loader(selected[start:],cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream,start=start):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];pred,state,info=model.step(ego,state,meta);assert all(bool(torch.isfinite(v).all()) for v in pred.values()) and info['audit_scratch_bytes']==0
            f=info['fine_integer'];re=info['readout_events'];logical=231067468800 if info['full_refresh'] else 105679929600
            stat=dict(core_fp32_conv_flops=logical-81749606400,fine_fp64_reference_flops=sum(z['actual_fp64_flops'] for z in f.values()),readout_fp64_reference_flops=sum(z['actual_fp64_conv_flops'] for z in re['layers']),fine_integer_macs=sum(z['actual_integer_macs'] for z in f.values()),readout_integer_macs=sum(z['actual_integer_macs'] for z in re['layers']),logical_core_conv_flops=logical,cache_bytes=info['cache_bytes'],reset=info['reset'],refresh=info['full_refresh'],layers=f,readout_events=re)
            old=gold[index]['arms']['fine_hybrid'];assert meta['frame']==gold[index]['frame'] and all(stat[k]==old[k] for k in stat)
            entry=dict(index=index,frame=meta['frame'],prediction_hash=tree_digest(pred),state_hash=tree_digest(state),statistics_hash=tree_digest(stat),statistics=stat);records.append(entry);log.write(json.dumps(entry)+'\n')
            if a.phase=='reference' and index==15:
                package=dict(identity=identity,next_index=16,state=state);torch.save(package,root/'state_at_016.pth');write_json(root/'checkpoint_metadata.json',dict(identity=identity,next_index=16,last_frame=meta['frame'],state_hash=entry['state_hash'],checkpoint_sha256=digest(root/'state_at_016.pth'),checkpoint_bytes=(root/'state_at_016.pth').stat().st_size,cache_bytes=info['cache_bytes']))
            write_json(root/'status.json',dict(status='RUNNING',phase=a.phase,scans=len(records),total=32-start))
    assert model_digest(model)==identity['model_digest'] and len(records)==32-start
    write_json(out/'report.json',dict(status='COMPLETED',pid=os.getpid(),scans=len(records),start_index=start,model_digest=identity['model_digest'],source_trajectory_exact=True,checkpoint_sha256=digest(root/'state_at_016.pth')))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','primitive-verification','calibration','implicit-verification','readout-calibration','fine-stream-verification','joint-verification','joint-source','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--phase',choices=('reference','resumed'));a=p.parse_args();a.validation_prefix=0;root=Path(a.output)
    if a.phase:
        out=root/a.phase;out.mkdir();phase_worker(a,out);return
    root.mkdir(parents=True,exist_ok=False);write_json(root/'status.json',dict(status='RUNNING',phase='initializing'));deadline=time.monotonic()+2700
    try:
        for phase in ('reference','resumed'):
            with (root/(phase+'_worker.log')).open('x') as log:
                result=subprocess.run([sys.executable,'-m','opencood.tools.check_int_fine_readout_restore',*sys.argv[1:],'--phase',phase],stdout=log,stderr=log,timeout=max(1,deadline-time.monotonic()))
            if result.returncode:raise RuntimeError(phase+' process failed: '+str(result.returncode))
        ref=[json.loads(x) for x in (root/'reference/frames.jsonl').read_text().splitlines()];res=[json.loads(x) for x in (root/'resumed/frames.jsonl').read_text().splitlines()];assert len(ref)==32 and len(res)==16 and ref[16:]==res and not res[0]['statistics']['reset'] and not res[0]['statistics']['refresh']
        rr=json.loads((root/'reference/report.json').read_text());sr=json.loads((root/'resumed/report.json').read_text());assert rr['pid']!=sr['pid'] and rr['model_digest']==sr['model_digest'];metadata=json.loads((root/'checkpoint_metadata.json').read_text())
        metrics=dict(parameters=34533972,**{k+'_mean':sum(x['statistics'][k] for x in ref)/32 for k in ('core_fp32_conv_flops','fine_fp64_reference_flops','readout_fp64_reference_flops','fine_integer_macs','readout_integer_macs','logical_core_conv_flops','cache_bytes')},model_step_ms_mean=None,ap50=None,ap70=None)
        write_json(root/'report.json',dict(status='COMPLETED',metrics=metrics,reference_pid=rr['pid'],resumed_pid=sr['pid'],reference_scans=32,resumed_scans=16,resume_index=16,checkpoint_sha256=metadata['checkpoint_sha256'],checkpoint_bytes=metadata['checkpoint_bytes'],prediction_state_statistics_exact=True,trained=False,ap_evaluated=False,timing_valid=False,boundary='Two separate worker processes, complete frozen task/fine/readout state saved to disk at16, resumed ordinary frame17 with no implicit reset and exact final16 prediction/state/statistics hashes. Both trajectories replay R209 operation/state ledger. No optimizer recovery, full-flow fault tolerance or AP claim.'))
        write_json(root/'status.json',dict(status='COMPLETED'))
    except Exception as exc:write_json(root/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
