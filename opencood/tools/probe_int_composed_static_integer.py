"""Joint candidate/encoder execution, with gated full static deployment audit."""
from opencood.tools.int_joint_adaptation_setup import *
from opencood.models.sub_modules.int_refresh_encoder_integer import ENCODER_PREFIXES
from opencood.models.sub_modules.int_candidate_integer import CANDIDATE_PREFIXES
from opencood.models.sub_modules.int_combined_integer import CombinedIntegerDetector
from opencood.models.sub_modules.int_composed_static_integer import ComposedStaticIntegerDetector


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
    candidate_proof=json.loads(Path(a.candidate_verification).read_text())
    assert candidate_proof['status']=='VERIFIED' and candidate_proof['run']=='R-242'
    candidate_source=Path(a.candidate_source)
    for n,h in candidate_proof['source_sha256'].items():assert digest(candidate_source/n)==h
    candidate_contract=json.loads((candidate_source/'contract.json').read_text())
    candidate_cal=json.loads((candidate_source/'calibration.json').read_text())
    for k in ('checkpoint_sha256','protected_checkpoint_sha256','initialization_sha256'):assert candidate_contract[k]==identity[k]
    assert tuple(candidate_contract['candidate_prefixes'])==CANDIDATE_PREFIXES and candidate_cal['relative_quantum']==.125 and candidate_cal['floor']==1e-4
    assert candidate_contract['evaluation_frames']==pc['evaluation_frames'] and candidate_contract['calibration_frames']==pc['calibration_frames']
    for filename,run,key,proof_path,scans in ((a.candidate_prefix_verification,'R-244','candidate',a.candidate_verification,512),(a.encoder_stream_verification,'R-246','encoder',a.encoder_verification,32)):
        gate=json.loads(Path(filename).read_text())
        assert gate['status']=='VERIFIED' and gate['run']==run and gate['scans']==scans and gate['checkpoint_sha256']==identity['checkpoint_sha256']
        assert gate[key+'_verification_sha256']==digest(Path(proof_path))
        assert all(gate['metrics'][n][key+'_integer_macs_mean']<gate['metrics']['dense'][key+'_fp64_reference_flops_mean']/2 for n in ('static','delta'))
    total=a.validation_prefix or 32
    selected=rows[:total] if a.validation_prefix else rows[512:544]
    assert len(selected)==total
    if a.validation_prefix:
        joint=json.loads(Path(a.joint_stream_verification).read_text())
        assert joint['status']=='VERIFIED' and joint['run']=='R-249' and joint['scans']==32
        assert joint['checkpoint_sha256']==identity['checkpoint_sha256']
        assert joint['branch_verification_sha256']=={g:digest(Path(getattr(a,g+'_verification'))) for g in ('candidate','encoder')}
        dense_mac=sum(joint['metrics']['dense'][g+'_fp64_reference_flops_mean'] for g in ('candidate','encoder'))/2
        assert all(sum(joint['metrics'][n][g+'_integer_macs_mean'] for g in ('candidate','encoder'))<dense_mac for n in ('static','delta'))
    else:
        assert [r['frame'] for r in selected]==pc['evaluation_frames'][:32]
    prefix_gold=None
    composed_gate=json.loads(Path(a.readout_static_verification).read_text());assert composed_gate['status']=='VERIFIED' and composed_gate['run']=='R-254' and composed_gate['scans']==512
    assert composed_gate['checkpoint_sha256']==identity['checkpoint_sha256']
    prefix=json.loads(Path(a.joint_prefix_verification).read_text());assert prefix['status']=='VERIFIED' and prefix['run']=='R-250' and prefix['scans']==512
    assert prefix['branch_verification_sha256']=={g:digest(Path(getattr(a,g+'_verification'))) for g in ('candidate','encoder')}
    prefix_source=Path(a.joint_prefix_source)
    for n,h in prefix['source_sha256'].items():assert digest(prefix_source/n)==h
    prefix_gold=[json.loads(x) for x in (prefix_source/'frames.jsonl').read_text().splitlines()];assert len(prefix_gold)==512
    assert not set(pc['calibration_frames'])&set(pc['evaluation_frames'])
    models=dict(control=FineReadoutIntegerDetector(source,cal['layers'],rc['quanta'],'hybrid',False).cuda().eval())
    modes=('static',)
    for mode in modes:
        models[mode]=CombinedIntegerDetector(source,cal['layers'],rc['quanta'],candidate_cal['layers'],cc['layers'],mode,mode,True).cuda().eval()
    models['composed']=ComposedStaticIntegerDetector(source,cal['layers'],rc['quanta'],candidate_cal['layers'],cc['layers'],True).cuda().eval()
    freeze([base]+list(models.values()));hashes={n:model_digest(m) for n,m in models.items()}
    contract=dict(**identity,candidate_verification_sha256=digest(Path(a.candidate_verification)),candidate_calibration_sha256=digest(candidate_source/'calibration.json'),candidate_prefix_verification_sha256=digest(Path(a.candidate_prefix_verification)),encoder_stream_verification_sha256=digest(Path(a.encoder_stream_verification)),candidate_prefixes=CANDIDATE_PREFIXES,encoder_verification_sha256=digest(Path(a.encoder_verification)),encoder_calibration_sha256=digest(prior/'calibration.json'),encoder_prefixes=ENCODER_PREFIXES,frames=[r['frame'] for r in selected],scans=total,frame_scope='validation_prefix' if a.validation_prefix else 'train32',joint_stream_verification_sha256=digest(Path(a.joint_stream_verification)) if a.validation_prefix else None,encoder_modes=['dense','static','delta'],fine_policy=FINE_POLICY,
        boundary='Fixed validation prefix; control and joint-static replay R250. Composed adds static sparse readout refresh to joint35 static, retaining fine hybrid and ordinary implicit reader. Frozen original R139/initrank128/quanta/P4. Exact paired prediction, task, fine/readout/candidate/encoder nA and storage. Typed arithmetic measured directly, no addition of marginal savings. No AP, timing, training or independent membrane claim.')
    contract.update(encoder_modes=['static','composed'],joint_prefix_verification_sha256=digest(Path(a.joint_prefix_verification)),readout_static_verification_sha256=digest(Path(a.readout_static_verification)))
    write_json(out/'contract.json',contract)
    states={n:None for n in models};base_state=None;records=[];_,stream=loader(selected,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            row=dict(frame=meta['frame'],arms={});preds={}
            for name,m in models.items():
                (pred,states[name],info),fp32=measured_step(m,x,states[name],meta);preds[name]=pred
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                values=ledger(m,info,fp32)
                for group in ('encoder','candidate'):
                    entries=info.get(group+'_integer',{})
                    values.update({group+'_layers':entries,group+'_fp64_reference_flops':sum(z['actual_fp64_flops'] for z in entries.values()),group+'_integer_macs':sum(z['actual_integer_macs'] for z in entries.values()),group+'_unit_spike_ac':sum(z['unit_spike_ac'] for z in entries.values()),group+'_cache_bytes':info.get(group+'_cache_bytes',0)})
                    values['logical_core_conv_flops']+=2*sum(z['dense_mac'] for z in entries.values())
                assert values['logical_core_conv_flops']==(231067468800 if info['full_refresh'] else 105679929600)
                row['arms'][name]=values
            gold=prefix_gold[index];assert row['frame']==gold['frame']
            for n in ('control','static'):assert row['arms'][n]==gold['arms'][n]
            for name in ('composed',):
                assert all(torch.equal(preds['static'][k],preds[name][k]) for k in preds[name])
                assert all(torch.equal(states['static'][k],states[name][k]) for k in base.state_keys)
                for key in ('readout','readout_accumulators'):assert all(torch.equal(a,b) and a.dtype==b.dtype for a,b in zip(states['static'][key],states[name][key]))
                for key,leaf in models['static'].core.observation.leaves().items():assert all(torch.equal(a,b) for a,b in zip(leaf.audit,models[name].core.observation.leaves()[key].audit))
                for group in ('encoder','candidate'):
                    for key,leaf in getattr(models['static'],group+'_leaves')().items():
                        other=getattr(models[name],group+'_leaves')()[key]
                        assert (leaf.audit is None)==(other.audit is None)
                        if leaf.audit is not None:assert all(torch.equal(a,b) for a,b in zip(leaf.audit,other.audit))
            assert row['arms']['static']['cache_bytes']==row['arms']['composed']['cache_bytes']
            for g in ('candidate','encoder'):
                for k in (g+'_layers',g+'_integer_macs',g+'_unit_spike_ac',g+'_fp64_reference_flops',g+'_cache_bytes'):assert row['arms']['static'][k]==row['arms']['composed'][k]
            assert row['arms']['composed']['readout_fp64_reference_flops']==0
            for name,m in models.items():
                if name!='control':m.clear_audit()
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=total))
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    result=metrics(models,records)
    for name,m in result.items():
        for key in tuple(g+'_'+k for g in ('encoder','candidate') for k in ('fp64_reference_flops','integer_macs','unit_spike_ac','cache_bytes')):
            m[key+'_mean']=sum(r['arms'][name][key] for r in records)/total
        m['cache_bytes_min']=min(r['arms'][name]['cache_bytes'] for r in records);m['cache_bytes_max']=max(r['arms'][name]['cache_bytes'] for r in records)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=result,weights_unchanged=True,integer_predictions_bit_exact=True,integer_task_states_exact=True,integer_inputs_accumulators_exact=True,trained=False,ap_evaluated=False,timing_valid=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ARGUMENTS+('encoder-verification','encoder-source','candidate-verification','candidate-source','candidate-prefix-verification','encoder-stream-verification'):p.add_argument('--'+n,required=True)
    p.add_argument('--readout-static-verification',required=True)
    p.add_argument('--validation-prefix',type=int,choices=(32,512),default=32);p.add_argument('--joint-stream-verification')
    for n in ('joint-prefix-verification','joint-prefix-source'):p.add_argument('--'+n)
    a=p.parse_args()
    if a.validation_prefix and not a.joint_stream_verification:p.error('Joint32 proof required for validation512')
    if not a.joint_prefix_verification or not a.joint_prefix_source:p.error('Composition requires verified joint512 source')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    budget=7200 if a.validation_prefix==512 else 2700
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('%dmin combined integer engineering'%(budget//60))));signal.alarm(budget)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
