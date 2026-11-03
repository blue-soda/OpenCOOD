"""Fixed four-arm quality attribution after separate and joint integer closure."""
from opencood.tools.int_joint_adaptation_setup import *
from opencood.models.sub_modules.int_candidate_integer import CandidateIntegerDetector,CANDIDATE_PREFIXES
from opencood.models.sub_modules.int_refresh_encoder_integer import RefreshEncoderIntegerDetector,ENCODER_PREFIXES
from opencood.models.sub_modules.int_combined_integer import CombinedIntegerDetector


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare(a,out)
    del exports
    branch_cal={};proof_hashes={}
    for group,run,prefixes in [('candidate','R-242',CANDIDATE_PREFIXES),('encoder','R-245',ENCODER_PREFIXES)]:
        proof_path=Path(getattr(a,group+'_verification'));v=json.loads(proof_path.read_text());prior=Path(getattr(a,group+'_source'))
        assert v['status']=='VERIFIED' and v['run']==run
        for name,h in v['source_sha256'].items():assert digest(prior/name)==h
        c=json.loads((prior/'contract.json').read_text());bc=json.loads((prior/'calibration.json').read_text())
        for k in ('checkpoint_sha256','protected_checkpoint_sha256','initialization_sha256'):assert c[k]==identity[k]
        assert tuple(c[group+'_prefixes'])==prefixes and bc['relative_quantum']==.125 and bc['floor']==1e-4
        branch_cal[group]=bc;proof_hashes[group]=digest(proof_path)
    gate=json.loads(Path(a.joint_stream_verification).read_text())
    assert gate['status']=='VERIFIED' and gate['run']=='R-249' and gate['scans']==32
    assert gate['checkpoint_sha256']==identity['checkpoint_sha256'] and gate['branch_verification_sha256']==proof_hashes
    for filename,run in ((a.resume_verification,'R-239'),(a.event_quality_verification,'R-236')):
        v=json.loads(Path(filename).read_text());assert v['status']=='VERIFIED' and v['run']==run
    qv=json.loads(Path(a.quality_verification).read_text());gp=Path(a.quality_source)/'ap_records.json'
    assert digest(gp)==qv['source_sha256']['ap_records.json'];gold=json.loads(gp.read_text())
    models=dict(control=FineReadoutIntegerDetector(source,cal['layers'],rc['quanta'],'hybrid',False).cuda().eval(),
                candidate=CandidateIntegerDetector(source,cal['layers'],rc['quanta'],branch_cal['candidate']['layers'],'dense',False).cuda().eval(),
                encoder=RefreshEncoderIntegerDetector(source,cal['layers'],rc['quanta'],branch_cal['encoder']['layers'],'dense',False).cuda().eval(),
                combined=CombinedIntegerDetector(source,cal['layers'],rc['quanta'],branch_cal['candidate']['layers'],branch_cal['encoder']['layers'],'dense','dense',False).cuda().eval())
    assert len(rows)==5794 and sum(r['supervised'] for r in rows)==1738;freeze([base]+list(models.values()))
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    contract=dict(**identity,branch_verification_sha256=proof_hashes,branch_calibration_sha256={g:digest(Path(getattr(a,g+'_source'))/'calibration.json') for g in branch_cal},joint_stream_verification_sha256=digest(Path(a.joint_stream_verification)),resume_verification_sha256=digest(Path(a.resume_verification)),event_quality_verification_sha256=digest(Path(a.event_quality_verification)),frames=[r['frame'] for r in rows],refresh_period=4,max_age=.45,
        boundary='Fixed four arms control/candidate/encoder/combined on full5794/1738. Original protected weights and initial rank128, original ANN-calibrated quanta, no adaptation, sweep or AP-based selection. Existing fine hybrid/implicit readout in all arms, newly covered layers dense FP64 references; this is a mixed execution reference, not all-dense or final deployment cost. R212 joint_dense raw AP must replay in control. Single quality GPU only after R236 and R239 finish, new joint32 proof required. No timing, interim AP or training; research validation, not independent test.')
    write_json(out/'contract.json',contract);dataset,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);assert all(bool(torch.isfinite(v).all()) for v in pred.values());values=ledger(m,info,fp32)
                for group in ('encoder','candidate'):
                    entries=info.get(group+'_integer',{})
                    values.update({group+'_layers':entries,group+'_fp64_reference_flops':sum(z['actual_fp64_flops'] for z in entries.values()),group+'_integer_macs':sum(z['actual_integer_macs'] for z in entries.values()),group+'_unit_spike_ac':sum(z['unit_spike_ac'] for z in entries.values()),group+'_cache_bytes':info.get(group+'_cache_bytes',0)})
                    values['logical_core_conv_flops']+=2*sum(z['dense_mac'] for z in entries.values())
                assert values['logical_core_conv_flops']==(231067468800 if info['full_refresh'] else 105679929600)
                assert values['audit_scratch_bytes']==0;row['arms'][n]=values
                if meta['supervised']:
                    batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch);boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            records.append(row);log.write(json.dumps(row)+'\n')
            if len(records)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=5794))
    assert len(records)==5794 and all(model_digest(m)==hashes[n] for n,m in models.items());write_json(out/'ap_records.json',ap)
    raw=json.loads((out/'ap_records.json').read_text());assert raw['control']==gold['joint_dense']
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,orders={n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in ap.items()}))
    result=metrics(models,records)
    for n,m in result.items():
        for g in ('candidate','encoder'):
            for k in ('fp64_reference_flops','integer_macs','unit_spike_ac','cache_bytes'):m[g+'_'+k+'_mean']=sum(r['arms'][n][g+'_'+k] for r in records)/5794
        m.update(ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0])
    write_json(out/'report.json',dict(status='COMPLETED',metrics=result,scans=5794,labels=1738,weights_unchanged=True,control_raw_ap_exact=True,trained=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ARGUMENTS+('candidate-verification','candidate-source','encoder-verification','encoder-source','joint-stream-verification','resume-verification','event-quality-verification'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=5794;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('480min fixed four-arm full quality')));signal.alarm(28800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
