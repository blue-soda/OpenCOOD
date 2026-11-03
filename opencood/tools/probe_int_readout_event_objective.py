"""Frozen learned-QAT exact-forward objective accounting against actual execution."""
from opencood.tools.int_joint_adaptation_setup import *
from opencood.models.sub_modules.int_readout_event_objective import EventObjectiveReadout
from opencood.tools.check_int_fine_readout_restore import tree_digest


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare(a,out)
    proof=json.loads(Path(a.timing_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-225' and proof['trained_checkpoint_sha256']==identity['trained_checkpoint_sha256']
    fp=Path(a.timing_source)/'frames.jsonl';assert digest(fp)==proof['source_sha256']['frames.jsonl'];gold=[json.loads(x) for x in fp.read_text().splitlines()][:256]
    selected=rows[512:768];assert len(selected)==len(gold)==256 and [x['frame'] for x in selected]==[x['frame'] for x in gold]
    model=JointStateQATDetector(exports['qat'],cal['layers'],rc['quanta'],True).cuda().eval()
    model.core.readout=EventObjectiveReadout(exports['qat'].core.readout,rc['quanta']).cuda().eval();freeze([base,model]);before=model_digest(model)
    contract=dict(**identity,timing_verification_sha256=digest(Path(a.timing_verification)),source_frames_sha256=digest(fp),frames=[x['frame'] for x in selected],cost_scope='Readout only; full-refresh inputs become new detached sending reference and contribute zero event cost. Dense refresh MAC remains separately charged. Exact weighted-MAC forward with derivative of1-exp(-abs(integer_delta)); exact unit-AC forward with abs derivative through quantizer STE. Previous integer reference detached, no pose warp. No optimizer, backward on real model, AP or timing.',boundary='Frozen diagnostic with additional all-layer int32 sending references; no new task memory or sparse deployment. Toy tests verify nonzero finite surrogate gradients; no claim of learned energy saving.')
    write_json(out/'contract.json',contract);records=[];state=None;_,stream=loader(selected,cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];g=gold[i]['arms']['joint'];model.core.readout.begin_frame(bool(g['refresh']))
            pred,state,info=model.step(ego,state,meta);assert info['full_refresh']==g['refresh'] and info['reset']==g['reset']
            prediction_hash=tree_digest(pred);task_hash=tree_digest({k:state[k] for k in base.state_keys});assert prediction_hash==g['prediction_hash'] and task_hash==g['task_hash']
            readout=model.core.readout;ls=readout.event_records;assert len(ls)==len(g['readout_events']['layers'])==4
            for x,y in zip(ls,g['readout_events']['layers']):assert x['weighted_event_mac']==y['actual_integer_macs'] and x['unit_spike_ac']==y['unit_spike_ac']
            weighted=int(readout.event_weighted);units=int(readout.event_units);assert weighted==g['readout_integer_macs'] and units==sum(x['unit_spike_ac'] for x in ls)
            logical=231067468800 if info['full_refresh'] else 105679929600;stat=ledger(model,info,logical-81749606400)
            record=dict(index=i,frame=meta['frame'],prediction_hash=prediction_hash,task_hash=task_hash,statistics=stat,objective_layers=ls,weighted_objective_mac=weighted,unit_objective_ac=units,objective_reference_bytes=readout.reference_bytes());records.append(record);log.write(json.dumps(record)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=i+1,total=256))
    assert len(records)==256 and model_digest(model)==before
    m=metrics({'objective_qat':model},[{'arms':{'objective_qat':r['statistics']}} for r in records])['objective_qat']
    m.update(readout_weighted_objective_mac_mean=sum(r['weighted_objective_mac'] for r in records)/256,readout_unit_objective_ac_mean=sum(r['unit_objective_ac'] for r in records)/256,objective_reference_bytes_mean=sum(r['objective_reference_bytes'] for r in records)/256)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=m,scans=256,weights_unchanged=True,predictions_tasks_exact=True,readout_event_costs_exact=True,trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ARGUMENTS+('timing-verification','timing-source'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min frozen event objective accounting')));signal.alarm(2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
