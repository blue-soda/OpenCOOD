"""Frozen own-state export for both event-training terminals, fixed train32."""
from opencood.tools.int_event_adaptation_setup import *


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare_event(a,out)
    proof=json.loads(Path(a.baseline_export_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-222'
    assert proof['trained_checkpoint_sha256']==identity['trained_checkpoint_sha256']
    gp=Path(a.baseline_export_source)/'frames.jsonl';assert digest(gp)==proof['source_sha256']['frames.jsonl'];gold=[json.loads(x) for x in gp.read_text().splitlines()]
    models={}
    for name,export in exports.items():
        models[name+'_layout']=JointStateQATDetector(export,cal['layers'],rc['quanta'],True).cuda().eval()
        for mode in ('dense','hybrid'):models[name+'_'+mode]=FineReadoutIntegerDetector(export,cal['layers'],rc['quanta'],mode,True).cuda().eval()
    selected=rows[512:544];assert len(selected)==len(gold)==32;freeze([base]+list(models.values()))
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(**identity,baseline_export_verification_sha256=digest(Path(a.baseline_export_verification)),frames=[x['frame'] for x in selected],policy=FINE_POLICY,boundary='Fixed train32, both R234389-update terminals, no selection/AP/timing/optimizer. Fresh-stream own states; training observer references are not deployment task memory. Same-weight QAT/dense/hybrid predictions and task states exact; integer n/A exact. Zero-coefficient control reconstructs original R221 model and R222 operation ledger. Actual event counts remeasured for event-trained weights.')
    write_json(out/'contract.json',contract);_,stream=loader(selected,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);preds[n]=pred
                assert all(bool(torch.isfinite(v).all()) for v in pred.values());row['arms'][n]=ledger(m,info,fp32)
            for name in exports:
                layout,dense,hybrid=[name+'_'+mode for mode in ('layout','dense','hybrid')]
                for right in (dense,hybrid):
                    assert all(torch.equal(preds[layout][k],preds[right][k]) for k in preds[layout])
                    assert all(torch.equal(states[layout][k],states[right][k]) for k in base.state_keys)
                assert torch.equal(states[dense]['readout'][0],states[hybrid]['readout'][0].to(states[dense]['readout'][0].dtype))
                assert all(torch.equal(x,y.to(x.dtype)) for x,y in zip(states[dense]['readout_accumulators'],states[hybrid]['readout_accumulators']))
                leaves=models[dense].core.observation.leaves()
                for key,leaf in models[hybrid].core.observation.leaves().items():assert all(torch.equal(x,y) for x,y in zip(leaves[key].audit,leaf.audit))
                models[dense].clear_audit();models[hybrid].clear_audit()
            assert row['frame']==gold[i]['frame']
            for mode,prior in (('layout','qat_layout'),('dense','joint_dense'),('hybrid','joint_hybrid')):assert row['arms']['control_'+mode]==gold[i]['arms'][prior]
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=32))
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics(models,records),scans=32,weights_unchanged=True,qat_export_exact=True,integer_inputs_accumulators_exact=True,control_original_ledger_exact=True,trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in EVENT_ARGUMENTS+('baseline-export-verification','baseline-export-source'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min fixed event export')));signal.alarm(3600)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
