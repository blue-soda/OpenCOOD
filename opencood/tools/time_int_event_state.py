"""Counterbalanced full-step timing of both fixed event-training terminals."""
from opencood.tools.int_event_adaptation_setup import *
from opencood.tools.time_int_periodic_compact import exclusive
from opencood.tools.check_int_fine_readout_restore import tree_digest
import os,time


def worker(a,out):
    uuid=os.environ['CUDA_VISIBLE_DEVICES'];exclusive(uuid)
    base,cfg,rows,source,exports,cal,rc,identity=prepare_event(a,out)
    export_hash=require_event_export(a,identity)
    bv=json.loads(Path(a.baseline_timing_verification).read_text())
    assert bv['status']=='VERIFIED' and bv['run']=='R-225' and bv['trained_checkpoint_sha256']==identity['trained_checkpoint_sha256']
    gp=Path(a.baseline_timing_source)/'frames.jsonl';assert digest(gp)==bv['source_sha256']['frames.jsonl']
    gold=[json.loads(x) for x in gp.read_text().splitlines()];assert len(gold)==512
    models={name:FineReadoutIntegerDetector(model,cal['layers'],rc['quanta'],'hybrid',False).cuda().eval() for name,model in exports.items()}
    selected=rows[512:768];assert len(selected)==256;freeze([base]+list(models.values()));hashes={n:model_digest(m) for n,m in models.items()};records=[]
    contract=dict(**identity,event_export_verification_sha256=export_hash,baseline_timing_verification_sha256=digest(Path(a.baseline_timing_verification)),frames=[r['frame'] for r in selected],uuid=uuid,device=torch.cuda.get_device_name(),torch_version=torch.__version__,passes=2,warmup_scans_per_pass=16,pass_mean_ratio_limit=1.15,policy=FINE_POLICY,boundary='GPU-exclusive each frame, host shared, two resident independent-state arms. Train256x2,16 warmups/pass,480 samples. Cyclic2 and reversed second pass. CUDA synchronize around complete model.step including Pillar/scatter/warp/readout/heads and built-in integer diagnostics; no Conv hooks/audit tensor capture. Loader/H2D/external hashes/exclusive checks excluded. Both fixed control/event actual joint hybrid terminals measured, no AP/learning or selection. Same-frame predictions/task hashes/operation ledgers must replay across passes; control also exactly replays R225 joint non-timing records. No newly measured ANN reference or AP. Any pass mean ratio>1.15 invalidates all timings without retry.')
    write_json(out/'contract.json',contract)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for repeat in range(2):
            states={n:None for n in models};_,stream=loader(selected,cfg,2,303)
            for i,sample in enumerate(stream):
                exclusive(uuid);ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];names=list(models);offset=i%len(names);order=names[offset:]+names[:offset]
                if repeat:order=list(reversed(order))
                row=dict(repeat=repeat,index=i,frame=meta['frame'],included=i>=16,execution_order=order,arms={})
                for n in order:
                    torch.cuda.synchronize();start=time.perf_counter();pred,states[n],info=models[n].step(ego,states[n],meta);torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
                    assert all(bool(torch.isfinite(v).all()) for v in pred.values()) and info.get('audit_scratch_bytes',0)==0
                    logical=231067468800 if info['full_refresh'] else 105679929600;fp32=logical-(81749606400 if 'fine_integer' in info else 0)
                    values=ledger(models[n],info,fp32);values.update(model_step_ms=ms,prediction_hash=tree_digest(pred),task_hash=tree_digest({k:states[n][k] for k in base.state_keys}));row['arms'][n]=values
                    if repeat:assert {k:v for k,v in values.items() if k!='model_step_ms'}=={k:v for k,v in records[i]['arms'][n].items() if k!='model_step_ms'}
                    if n=='control':
                        prior=gold[repeat*256+i];assert prior['frame']==meta['frame']
                        assert {k:v for k,v in values.items() if k!='model_step_ms'}=={k:v for k,v in prior['arms']['joint'].items() if k!='model_step_ms'}
                records.append(row);log.write(json.dumps(row)+'\n')
                if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='timing',repeat=repeat,scans=i+1,total=256))
    exclusive(uuid);assert len(records)==512 and all(model_digest(m)==hashes[n] for n,m in models.items());chosen=[r for r in records if r['included']];assert len(chosen)==480;result=metrics(models,chosen)
    for n,m in result.items():
        values=[r['arms'][n]['model_step_ms'] for r in chosen];pp=[float(np.mean([r['arms'][n]['model_step_ms'] for r in chosen if r['repeat']==i])) for i in (0,1)]
        phases={}
        for name,refresh in [('refresh',True),('ordinary',False)]:
            z=[r['arms'][n]['model_step_ms'] for r in chosen if bool(r['arms'][n]['refresh'])==refresh];phases[name]=dict(samples=len(z),model_step_ms_mean=float(np.mean(z)))
        m.update(samples=480,model_step_ms_mean=float(np.mean(values)),model_step_ms_p50=float(np.median(values)),model_step_ms_p95=float(np.percentile(values,95)),per_pass_ms=pp,pass_mean_ratio=max(pp)/min(pp),phase_metrics=phases)
    valid=all(m['pass_mean_ratio']<=1.15 for m in result.values());write_json(out/'report.json',dict(status='COMPLETED',timing_valid=valid,metrics=result,weights_unchanged=True,repeat_predictions_tasks_ledgers_exact=True,control_original_ledger_exact=True,trained=False,ap_evaluated=False));write_json(out/'status.json',dict(status='COMPLETED',timing_valid=valid))


def main():
    p=argparse.ArgumentParser()
    for n in EVENT_ARGUMENTS+('event-export-verification','baseline-timing-verification','baseline-timing-source'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min paired event-terminal timing')));signal.alarm(3600)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False));raise
if __name__=='__main__':main()
