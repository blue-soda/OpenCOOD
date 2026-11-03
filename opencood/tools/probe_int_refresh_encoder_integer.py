"""Fixed32 closed-loop encoder integer execution after a frozen shadow audit."""
from opencood.tools.int_joint_adaptation_setup import *
from opencood.models.sub_modules.int_refresh_encoder_integer import RefreshEncoderIntegerDetector,ENCODER_PREFIXES


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare(a,out)
    del exports
    proof=json.loads(Path(a.encoder_verification).read_text())
    assert proof['status']=='VERIFIED' and proof['run']=='R-245'
    prior=Path(a.encoder_source)
    for n,h in proof['source_sha256'].items():assert digest(prior/n)==h
    pc=json.loads((prior/'contract.json').read_text());cc=json.loads((prior/'calibration.json').read_text())
    for k in ('checkpoint_sha256','protected_checkpoint_sha256','initialization_sha256'):assert pc[k]==identity[k]
    assert tuple(pc['encoder_prefixes'])==ENCODER_PREFIXES and cc['relative_quantum']==.125 and cc['floor']==1e-4
    total=a.validation_prefix or 32
    selected=rows[:total] if a.validation_prefix else rows[512:544]
    assert len(selected)==total
    if a.validation_prefix:
        gate=json.loads(Path(a.encoder_stream_verification).read_text())
        assert gate['status']=='VERIFIED' and gate['run']=='R-246' and gate['scans']==32
        assert gate['checkpoint_sha256']==identity['checkpoint_sha256'] and gate['encoder_verification_sha256']==digest(Path(a.encoder_verification))
        # Arithmetic admission only: require each actual integer implementation
        # to use fewer encoder MACs than its matched dense reference. No AP gate.
        dense_mac=gate['metrics']['dense']['encoder_fp64_reference_flops_mean']/2
        assert all(gate['metrics'][n]['encoder_integer_macs_mean']<dense_mac for n in ('static','delta'))
    else:
        assert [r['frame'] for r in selected]==pc['evaluation_frames'][:32]
    assert not set(pc['calibration_frames'])&set(pc['evaluation_frames'])
    models=dict(control=FineReadoutIntegerDetector(source,cal['layers'],rc['quanta'],'hybrid',False).cuda().eval())
    for mode in ('dense','static','delta'):
        models[mode]=RefreshEncoderIntegerDetector(source,cal['layers'],rc['quanta'],cc['layers'],mode,True).cuda().eval()
    freeze([base]+list(models.values()));hashes={n:model_digest(m) for n,m in models.items()}
    contract=dict(**identity,encoder_verification_sha256=digest(Path(a.encoder_verification)),encoder_calibration_sha256=digest(prior/'calibration.json'),encoder_stream_verification_sha256=digest(Path(a.encoder_stream_verification)) if a.validation_prefix else None,encoder_prefixes=ENCODER_PREFIXES,frames=[r['frame'] for r in selected],scans=total,frame_scope='validation_prefix' if a.validation_prefix else 'train32',encoder_modes=['dense','static','delta'],fine_policy=FINE_POLICY,
        boundary=('Fixed validation-prefix512' if a.validation_prefix else 'Fixed train512:544')+' frozen own-state streams. Existing hybrid fine/implicit readout in all arms; only20 full-refresh encoder Conv inputs/weights newly quantized. Dense FP64, real static integer and real delta integer encoders must agree exactly in task/prediction/n/A. Original ANN-derived calibration frozen, no policy selection. Gates, protected task updates, nonlinearities and warp unchanged. Encoder references survive skipped ordinary frames and reset on scene/gap/start. No AP, timing or training, no independent task membrane advantage. Source training-adapted models are not used; task and execution state distinct. Audit scratch reported separately from deployment state.')
    write_json(out/'contract.json',contract)
    states={n:None for n in models};base_state=None;records=[];_,stream=loader(selected,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample in stream:
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            row=dict(frame=meta['frame'],arms={});preds={}
            for name,m in models.items():
                (pred,states[name],info),fp32=measured_step(m,x,states[name],meta);preds[name]=pred
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                encoder=info.get('encoder_integer',{});values=ledger(m,info,fp32)
                values.update(encoder_layers=encoder,encoder_fp64_reference_flops=sum(z['actual_fp64_flops'] for z in encoder.values()),encoder_integer_macs=sum(z['actual_integer_macs'] for z in encoder.values()),encoder_unit_spike_ac=sum(z['unit_spike_ac'] for z in encoder.values()),encoder_cache_bytes=info.get('encoder_cache_bytes',0))
                values['logical_core_conv_flops']+=2*sum(z['dense_mac'] for z in encoder.values())
                assert values['logical_core_conv_flops']==(231067468800 if info['full_refresh'] else 105679929600)
                row['arms'][name]=values
            for name in ('static','delta'):
                assert all(torch.equal(preds['dense'][k],preds[name][k]) for k in preds[name])
                assert all(torch.equal(states['dense'][k],states[name][k]) for k in base.state_keys)
                for key,leaf in models['dense'].encoder_leaves().items():
                    other=models[name].encoder_leaves()[key]
                    assert (leaf.audit is None)==(other.audit is None)
                    if leaf.audit is not None:assert all(torch.equal(a,b) for a,b in zip(leaf.audit,other.audit))
            for name,m in models.items():
                if name!='control':m.clear_audit()
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=total))
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    result=metrics(models,records)
    for name,m in result.items():
        for key in ('encoder_fp64_reference_flops','encoder_integer_macs','encoder_unit_spike_ac','encoder_cache_bytes'):
            m[key+'_mean']=sum(r['arms'][name][key] for r in records)/total
        m['cache_bytes_min']=min(r['arms'][name]['cache_bytes'] for r in records);m['cache_bytes_max']=max(r['arms'][name]['cache_bytes'] for r in records)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=result,weights_unchanged=True,integer_predictions_bit_exact=True,integer_task_states_exact=True,integer_inputs_accumulators_exact=True,trained=False,ap_evaluated=False,timing_valid=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ARGUMENTS+('encoder-verification','encoder-source'):p.add_argument('--'+n,required=True)
    p.add_argument('--validation-prefix',type=int,choices=(0,512),default=0);p.add_argument('--encoder-stream-verification')
    a=p.parse_args()
    if a.validation_prefix and not a.encoder_stream_verification:p.error('Encoder stream verification required for validation-prefix512')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    budget=5400 if a.validation_prefix else 2700
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('%dmin encoder integer engineering'%(budget//60))));signal.alarm(budget)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
