"""Four fixed complete-stream arms after matched ordinary-state adaptation."""
from opencood.tools.int_joint_adaptation_setup import *


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare(a,out)
    ev=json.loads(Path(a.export_verification).read_text());assert ev['status']=='VERIFIED' and ev['run']=='R-222' and ev['trained_checkpoint_sha256']==identity['trained_checkpoint_sha256']
    assert ev['calibration_sha256']==identity['calibration_sha256'] and ev['readout_calibration_sha256']==identity['readout_calibration_sha256']
    qv=json.loads(Path(a.quality_verification).read_text());gp=Path(a.quality_source)/'ap_records.json';assert digest(gp)==qv['source_sha256']['ap_records.json'];gold=json.loads(gp.read_text())
    models=dict(frozen_ann=source,frozen_qat=JointStateQATDetector(source,cal['layers'],rc['quanta'],True).cuda().eval(),ann=exports['ann'],qat=JointStateQATDetector(exports['qat'],cal['layers'],rc['quanta'],True).cuda().eval())
    assert len(rows)==5794 and sum(r['supervised'] for r in rows)==1738;freeze([base]+list(models.values()))
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    contract=dict(**identity,export_verification_sha256=digest(Path(a.export_verification)),frames=[r['frame'] for r in rows],boundary='All four fixed full5794/1738 arms: frozen ANN, frozen joint QAT, fixed ANN adaptation, fixed joint QAT adaptation. Both389-update terminals retained irrespective of train256 AP. No training/quantum/epoch selection. Frozen raw AP replay R212. FP64 exact training-reference forward, task-state-only layout22.5792MB is not sparse deployment cost. No model latency or independent test claim.')
    write_json(out/'contract.json',contract);dataset,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);assert all(bool(torch.isfinite(v).all()) for v in pred.values());row['arms'][n]=ledger(m,info,fp32)
                if meta['supervised']:
                    batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch);boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            records.append(row);log.write(json.dumps(row)+'\n')
            if len(records)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=5794))
    assert len(records)==5794 and all(model_digest(m)==hashes[n] for n,m in models.items());write_json(out/'ap_records.json',ap)
    raw=json.loads((out/'ap_records.json').read_text());assert raw['frozen_ann']==gold['control'] and raw['frozen_qat']==gold['joint_dense']
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,orders={n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in ap.items()}))
    result=metrics(models,records)
    for n,m in result.items():m.update(ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0])
    write_json(out/'report.json',dict(status='COMPLETED',metrics=result,scans=5794,labels=1738,weights_unchanged=True,frozen_raw_ap_exact=True,trained=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ARGUMENTS+('export-verification',):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=5794;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240min joint state full quality')));signal.alarm(14400)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
