"""Frozen composition: complete D recurrence and local expensive readout reuse."""
import json
from pathlib import Path
import torch
from opencood.models.sub_modules.int_recurrent_deep import RecurrentDeepDetector
from opencood.models.sub_modules.int_local_recurrent_deep import LocalRecurrentDeepDetector
from opencood.models.sub_modules.int_local_readout import LocalReadoutCache
from opencood.tools.launch_int_memory import digest
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.train_int_recurrent_deep import cost_of,evaluate


def compare_composition(a,cfg,train,val,baseline,out,costs,spatial_macs,repair,checkpoint,reference,source,expected):
    calibration_path=Path(a.recurrent_readout_calibration)
    calibration=json.loads(calibration_path.read_text())
    assert calibration['status']=='COMPLETED' and calibration['split']=='train'
    assert calibration['scans']==512 and calibration['fit']['status']=='FITTED'
    assert calibration['base_threshold']==.5 and calibration['target_ratio']==.8
    assert calibration['exact_replay_max_error']==0
    assert calibration['identity']=={k:source[k] for k in calibration['identity']}
    threshold=calibration['fit']['threshold'];period=a.recurrent_local_period
    modes=dict(dense_recurrent=RecurrentDeepDetector(baseline,repair,period),
        local_d=LocalRecurrentDeepDetector(baseline,repair,period),
        local_readout=RecurrentDeepDetector(baseline,repair,period),
        joint_selected=LocalRecurrentDeepDetector(baseline,repair,period),
        joint_masked_dense=LocalRecurrentDeepDetector(baseline,repair,period,'masked_dense'))
    for name in ('local_readout','joint_selected','joint_masked_dense'):
        execution='masked_dense' if name=='joint_masked_dense' else 'selected'
        modes[name].readout_cache=LocalReadoutCache(baseline.readout,baseline.bounds,threshold,execution,3)
    contract=dict(checkpoint=checkpoint,reference_sha256=digest(reference),deep_period=period,readout_period=3,
        calibration=dict(path=str(calibration_path),sha256=digest(calibration_path),threshold=threshold,evidence=calibration),
        scope='Frozen composition, no training or threshold refit. All modes own FM/H/D/readout/confidence trajectories.',
        selection='D uses own actual H update support plus uncovered D. Readout compares actual current shallow+corrected D input to own retained reference, with existing confidence/uncovered protection. Do not substitute upstream H mask for readout input changes.',
        refresh='D period4/.35s by default; readout independently period3/.25s. FM reset resets both. No additional readout refresh on D full frames: actual new readout input drives existing selector.',
        controls='Original baseline; dense D; local D only; local readout only; joint selected; joint dense-compute masked policy. Separate compression/longer D refresh benefits from local reuse.',
        accounting='Actual convolution and functional H/D/readout MACs x2; include persistent output/reference/confidence and packed weights separately. Timing includes selection, warp, gather and scatter; discarded replays excluded.',
        limits='Readout reuse adds persistent cache and previously reduced FLOPs but increased GPU time. Composition may accumulate errors; no assumed additive speedup or AP preservation.')
    write_json(out/'recurrent_readout_contract.json',contract)
    selected_names=('local_d','local_readout','joint_selected')
    def replay(name,ego,previous,meta,pred,state,info):
        mode=modes[name];deep_execution=getattr(mode.deep,'execution',None)
        cache=mode.readout_cache;cache_execution=cache.execution if cache else None
        hmask=info.get('spatial_update_mask');rmask=cache.last_mask.clone() if cache else None
        if deep_execution is not None:mode.deep.execution='masked_dense'
        if cache:cache.execution='masked_dense'
        try:other,other_state,other_info=mode.step(ego,previous,meta)
        finally:
            if deep_execution is not None:mode.deep.execution=deep_execution
            if cache:cache.execution=cache_execution
        if hmask is not None:assert torch.equal(hmask,other_info['spatial_update_mask'])
        if cache:assert torch.equal(rmask,cache.last_mask)
        errors=[float((pred[k]-other[k]).abs().max()) for k in pred]
        for k in pred:assert torch.allclose(pred[k],other[k],atol=1e-3,rtol=1e-3)
        left,right=state['deep']['value'],other_state['deep']['value']
        assert torch.allclose(left,right,atol=1e-4,rtol=1e-4)
        readout_error=0.
        if cache:
            left_readout,right_readout=state['readout']['value'],other_state['readout']['value']
            assert torch.allclose(left_readout,right_readout,atol=1e-3,rtol=1e-3)
            readout_error=float((left_readout-right_readout).abs().max())
        return dict(mode=name,prediction_max_error=max(errors),deep_max_error=float((left-right).abs().max()),readout_max_error=readout_error)
    states={k:None for k in modes};records=[];d_partial=0;r_partial=0;period1_max=0.
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            for name,mode in modes.items():
                previous=states[name];counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.repair]))
                try:pred,states[name],info=mode.step(ego,previous,meta)
                finally:counter.close()
                actual=counter.total+sum(info.get(k,0) for k in ('spatial_extra_linear_macs','deep_extra_linear_macs','readout_extra_linear_macs'))
                assert actual==cost_of(costs,spatial_macs,info,True),(name,actual,cost_of(costs,spatial_macs,info,True))
                if name in selected_names:records.append(replay(name,ego,previous,meta,pred,states[name],info))
                if name=='joint_selected':
                    d_partial+=int(info.get('deep_macs_saved',0)>0);r_partial+=int(info.get('readout_macs_saved',0)>0)
                if name=='dense_recurrent':
                    mode.readout_cache=LocalReadoutCache(baseline.readout,baseline.bounds,threshold,period=1)
                    try:other,_,_=mode.step(ego,None if previous is None else dict(previous,readout=None),meta)
                    finally:mode.readout_cache=None
                    for k in pred:
                        period1_max=max(period1_max,float((pred[k]-other[k]).abs().max()))
                        assert torch.equal(pred[k],other[k]),'Period1 readout changed dense recurrence'
    assert d_partial>0 and r_partial>0
    write_json(out/'recurrent_readout_engineering.json',dict(status='PASSED',frames=32,actual_macs_verified=True,
        d_sparse_frames=d_partial,readout_sparse_frames=r_partial,period1_max_error=period1_max,
        max_prediction_error=max(r['prediction_max_error'] for r in records),
        max_readout_error=max(r['readout_max_error'] for r in records),max_deep_error=max(r['deep_max_error'] for r in records)))
    audits=[]
    def audit(name,ego,previous,meta,pred,state,info,index):
        if name in selected_names and index%200==0:
            result=replay(name,ego,previous,meta,pred,state,info);result['scan']=index+1;audits.append(result)
            write_json(out/'recurrent_readout_replays.json',audits)
    directory=out/'full';directory.mkdir()
    result=evaluate(a,baseline,None,cfg,val,directory,costs,spatial_macs,'composition',variants=modes,
        expected_ap={'dense_recurrent':expected},audit=audit,
        note='Frozen D/readout composition. Six independent histories rotate/reverse on one GPU. Same-input audits outside model timer. Conv/functional-linear MACsx2; no assumed latency benefit. Readout threshold transferred unchanged from train-only fit.')
    result['recurrent_readout_contract']=contract;result['same_input_replays']=len(audits)
    write_json(out/'report.json',result);write_json(out/'status.json',dict(status='COMPLETED',scans=len(val)))
