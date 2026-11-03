"""Gated fixed-prefix equal-quantization replacement of readout initialization."""
from opencood.tools.int_joint_adaptation_setup import *
from opencood.models.sub_modules.int_static_refresh_readout import StaticRefreshReadoutDetector


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare(a,out)
    del exports
    gate=json.loads(Path(a.joint_verification).read_text())
    assert gate['status']=='VERIFIED' and gate['run']==('R-214' if a.validation_prefix else 'R-209')
    for key in ('checkpoint_sha256','calibration_sha256','readout_calibration_sha256'):assert gate[key]==identity[key]
    prior=Path(a.joint_source)
    for n,h in gate['source_sha256'].items():assert digest(prior/n)==h
    gold=[json.loads(x) for x in (prior/'frames.jsonl').read_text().splitlines()]
    total=a.validation_prefix or 32
    if a.validation_prefix:assert len(gold)==5794;gold=gold[:total]
    selected=rows[:total] if a.validation_prefix else rows[512:544];assert len(selected)==len(gold)==total
    if total>=512:
        prefix=json.loads(Path(a.prefix_verification).read_text())
        assert prefix['status']=='VERIFIED' and prefix['run']==('R-254' if total==5794 else 'R-253') and prefix['scans']==(512 if total==5794 else 32)
        assert prefix['joint_verification_sha256']==digest(Path(a.joint_verification))
        for key in ('checkpoint_sha256','calibration_sha256','readout_calibration_sha256'):assert prefix[key]==identity[key]
    assert [r['frame'] for r in selected]==[r['frame'] for r in gold]
    models=dict(control=FineReadoutIntegerDetector(source,cal['layers'],rc['quanta'],'hybrid',True).cuda().eval(),static_refresh=StaticRefreshReadoutDetector(source,cal['layers'],rc['quanta'],True).cuda().eval())
    freeze([base]+list(models.values()));hashes={n:model_digest(m) for n,m in models.items()}
    contract=dict(**identity,joint_verification_sha256=digest(Path(a.joint_verification)),prefix_verification_sha256=digest(Path(a.prefix_verification)) if total>=512 else None,frames=[r['frame'] for r in selected],scans=total,frame_scope='validation_prefix%d'%total if a.validation_prefix else 'train512:544',refresh_period=source.period,max_age=source.max_age,
        boundary=('Fixed validation first%d, control replays R214. '%total if a.validation_prefix else 'Fixed train512:544, control replays R209. ')+'Original R139 protected weights/initial rank128 and ANN quanta. Static refresh initializes readout A from sparse current integer inputs instead of dense FP64 reference; ordinary delta path unchanged. Exact outputs/task/fine n-A/readout n-A and storage required. No new task membrane, training, AP or timing; learned exports loaded for provenance then discarded.')
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
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=total))
    assert len(records)==total and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics(models,records),weights_unchanged=True,predictions_bit_exact=True,task_states_exact=True,integer_states_exact=True,storage_exact=True,control_replayed=True,trained=False,ap_evaluated=False,timing_valid=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ARGUMENTS+('joint-verification','joint-source'):p.add_argument('--'+n,required=True)
    p.add_argument('--validation-prefix',type=int,choices=(0,32,512,5794),default=0);p.add_argument('--prefix-verification')
    a=p.parse_args()
    if a.validation_prefix>=512 and not a.prefix_verification:p.error('Verified shorter prefix required for longer stream')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    budget=21600 if a.validation_prefix==5794 else 5400 if a.validation_prefix==512 else 2700
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('%dmin static refresh prefix'%(budget//60))));signal.alarm(budget)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
