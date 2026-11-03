"""Full frozen learned-QAT dense/hybrid execution, independently of full AP."""
from opencood.tools.int_joint_adaptation_setup import *


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare(a,out)
    ev=json.loads(Path(a.export_verification).read_text());assert ev['status']=='VERIFIED' and ev['run']=='R-222' and ev['trained_checkpoint_sha256']==identity['trained_checkpoint_sha256']
    assert ev['calibration_sha256']==identity['calibration_sha256'] and ev['readout_calibration_sha256']==identity['readout_calibration_sha256']
    models=dict(control=source,joint_dense=FineReadoutIntegerDetector(exports['qat'],cal['layers'],rc['quanta'],'dense',True).cuda().eval(),joint_hybrid=FineReadoutIntegerDetector(exports['qat'],cal['layers'],rc['quanta'],'hybrid',True).cuda().eval())
    selected=rows;assert len(selected)==5794;freeze([base]+list(models.values()))
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(**identity,export_verification_sha256=digest(Path(a.export_verification)),frames=[r['frame'] for r in selected],policy=FINE_POLICY,boundary='Full5794 learned QAT dense/hybrid own-state prediction/task/n-A equivalence. All source identities gated by R221 and R222; fixed learned389-update terminal, no AP/optimizer/timing or candidate selection. Untrained ANN is an operation-count control, not a task-state equality target. Full quality R223 is independent; no borrowed old sparse event ledger or old AP. Fixed fine/readout weights/quanta; new event ledger measured after adaptation. Fine n/A audit transient, implicit readout persistent state counted.')
    write_json(out/'contract.json',contract);_,stream=loader(selected,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);preds[n]=pred
                assert all(bool(torch.isfinite(v).all()) for v in pred.values());row['arms'][n]=ledger(m,info,fp32)
            for left,right in (('joint_dense','joint_hybrid'),):
                assert all(torch.equal(preds[left][k],preds[right][k]) for k in preds[left])
                assert all(torch.equal(states[left][k],states[right][k]) for k in base.state_keys)
            assert all(row['arms'][n]['logical_core_conv_flops']==row['arms']['control']['logical_core_conv_flops'] for n in models)
            assert torch.equal(states['joint_dense']['readout'][0],states['joint_hybrid']['readout'][0].to(states['joint_dense']['readout'][0].dtype))
            assert all(torch.equal(x,y.to(x.dtype)) for x,y in zip(states['joint_dense']['readout_accumulators'],states['joint_hybrid']['readout_accumulators']))
            leaves=models['joint_dense'].core.observation.leaves()
            for n,leaf in models['joint_hybrid'].core.observation.leaves().items():assert all(torch.equal(x,y) for x,y in zip(leaves[n].audit,leaf.audit))
            for n in ('joint_dense','joint_hybrid'):models[n].clear_audit()
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=5794))
    assert len(records)==5794 and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics(models,records),scans=5794,weights_unchanged=True,qat_export_exact=True,integer_inputs_accumulators_exact=True,trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ARGUMENTS+('export-verification',):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=5794;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('360min learned joint full execution')));signal.alarm(21600)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
