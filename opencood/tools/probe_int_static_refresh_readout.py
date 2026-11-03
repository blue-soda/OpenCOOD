"""Fixed32 equal-quantization replacement of readout refresh initialization."""
from opencood.tools.int_joint_adaptation_setup import *
from opencood.models.sub_modules.int_static_refresh_readout import StaticRefreshReadoutDetector


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare(a,out)
    del exports
    gate=json.loads(Path(a.joint_verification).read_text())
    assert gate['status']=='VERIFIED' and gate['run']=='R-209'
    for key in ('checkpoint_sha256','calibration_sha256','readout_calibration_sha256'):assert gate[key]==identity[key]
    prior=Path(a.joint_source)
    for n,h in gate['source_sha256'].items():assert digest(prior/n)==h
    gold=[json.loads(x) for x in (prior/'frames.jsonl').read_text().splitlines()]
    selected=rows[512:544];assert len(selected)==len(gold)==32
    assert [r['frame'] for r in selected]==[r['frame'] for r in gold]
    models=dict(control=FineReadoutIntegerDetector(source,cal['layers'],rc['quanta'],'hybrid',True).cuda().eval(),static_refresh=StaticRefreshReadoutDetector(source,cal['layers'],rc['quanta'],True).cuda().eval())
    freeze([base]+list(models.values()));hashes={n:model_digest(m) for n,m in models.items()}
    contract=dict(**identity,joint_verification_sha256=digest(Path(a.joint_verification)),frames=[r['frame'] for r in selected],scans=32,refresh_period=source.period,max_age=source.max_age,
        boundary='Fixed train512:544, original R139 protected weights/initial rank128 and ANN quanta. Control hybrid fine and implicit readout replays R209. Static refresh initializes readout A from sparse current integer inputs instead of dense FP64 reference; ordinary delta path unchanged. Exact outputs/task/fine n-A/readout n-A and storage required. No new task membrane, training, AP or timing; learned exports loaded for provenance then discarded.')
    assert source.period==4 and source.max_age==.45
    write_json(out/'contract.json',contract)
    states={n:None for n in models};base_state=None;records=[];_,stream=loader(selected,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            row=dict(frame=meta['frame'],arms={});preds={}
            for name,m in models.items():
                (pred,states[name],info),fp32=measured_step(m,x,states[name],meta);preds[name]=pred
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                row['arms'][name]=ledger(m,info,fp32)
            assert row['arms']['control']==gold[index]['arms']['fine_hybrid']
            assert all(torch.equal(preds['control'][k],preds['static_refresh'][k]) for k in preds['control'])
            assert all(torch.equal(states['control'][k],states['static_refresh'][k]) for k in base.state_keys)
            for key in ('readout','readout_accumulators'):
                assert all(torch.equal(a,b) and a.dtype==b.dtype for a,b in zip(states['control'][key],states['static_refresh'][key]))
            ref=models['control'].core.observation.leaves()
            for key,leaf in models['static_refresh'].core.observation.leaves().items():assert all(torch.equal(a,b) for a,b in zip(ref[key].audit,leaf.audit))
            old=row['arms']['control'];new=row['arms']['static_refresh']
            for key in set(old)-{'readout_events','readout_fp64_reference_flops','readout_integer_macs'}:assert old[key]==new[key]
            assert new['readout_fp64_reference_flops']==0
            for old_layer,new_layer in zip(old['readout_events']['layers'],new['readout_events']['layers']):
                assert new_layer['dense_reference_calls']==0 and new_layer['actual_integer_macs']==new_layer['weighted_event_mac']<=new_layer['dense_mac']
                if not old['refresh']:assert old_layer==new_layer
            if not old['refresh']:assert old['readout_events']==new['readout_events']
            for m in models.values():m.clear_audit()
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=32))
    assert len(records)==32 and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics(models,records),weights_unchanged=True,predictions_bit_exact=True,task_states_exact=True,integer_states_exact=True,storage_exact=True,control_replayed=True,trained=False,ap_evaluated=False,timing_valid=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ARGUMENTS+('joint-verification','joint-source'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=0
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min static refresh fixed32')));signal.alarm(2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
