"""Actual full-stream execution of the fixed event-regularized terminal."""
from opencood.tools.int_event_adaptation_setup import *


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare_event(a,out);export_hash=require_event_export(a,identity)
    bv=json.loads(Path(a.baseline_execution_verification).read_text());assert bv['status']=='VERIFIED' and bv['run']=='R-224' and bv['trained_checkpoint_sha256']==identity['trained_checkpoint_sha256']
    gp=Path(a.baseline_execution_source)/'frames.jsonl';assert digest(gp)==bv['source_sha256']['frames.jsonl'];gold=[json.loads(x) for x in gp.read_text().splitlines()]
    models=dict(control=source,joint_dense=FineReadoutIntegerDetector(exports['event'],cal['layers'],rc['quanta'],'dense',True).cuda().eval(),joint_hybrid=FineReadoutIntegerDetector(exports['event'],cal['layers'],rc['quanta'],'hybrid',True).cuda().eval())
    assert len(rows)==len(gold)==5794;freeze([base]+list(models.values()))
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(**identity,event_export_verification_sha256=export_hash,baseline_execution_verification_sha256=digest(Path(a.baseline_execution_verification)),frames=[r['frame'] for r in rows],policy=FINE_POLICY,boundary='Full5794 actual event-trained dense/hybrid own-state prediction/task/n-A equality. Original untrained ANN is an operation-count control only. Fixed event terminal from R234, no AP/optimizer/timing. Frozen fine layers and ANN operation ledger replay R224; newly measured readout events, no borrowing training-time event averages or detection-only event cost. Audit shadow buffers separated from persistent state.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);preds[n]=pred
                assert all(bool(torch.isfinite(v).all()) for v in pred.values());row['arms'][n]=ledger(m,info,fp32)
            assert all(torch.equal(preds['joint_dense'][k],preds['joint_hybrid'][k]) for k in preds['joint_dense'])
            assert all(torch.equal(states['joint_dense'][k],states['joint_hybrid'][k]) for k in base.state_keys)
            assert torch.equal(states['joint_dense']['readout'][0],states['joint_hybrid']['readout'][0].to(states['joint_dense']['readout'][0].dtype))
            assert all(torch.equal(x,y.to(x.dtype)) for x,y in zip(states['joint_dense']['readout_accumulators'],states['joint_hybrid']['readout_accumulators']))
            leaves=models['joint_dense'].core.observation.leaves()
            for key,leaf in models['joint_hybrid'].core.observation.leaves().items():assert all(torch.equal(x,y) for x,y in zip(leaves[key].audit,leaf.audit))
            for n in ('joint_dense','joint_hybrid'):models[n].clear_audit()
            assert row['frame']==gold[i]['frame'] and row['arms']['control']==gold[i]['arms']['control']
            for n in ('joint_dense','joint_hybrid'):assert row['arms'][n]['layers']==gold[i]['arms'][n]['layers']
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=5794))
    assert len(records)==5794 and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics(models,records),scans=5794,weights_unchanged=True,qat_export_exact=True,integer_inputs_accumulators_exact=True,trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in EVENT_ARGUMENTS+('event-export-verification','baseline-execution-verification','baseline-execution-source'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=5794;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('360min fixed event full execution')));signal.alarm(21600)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
