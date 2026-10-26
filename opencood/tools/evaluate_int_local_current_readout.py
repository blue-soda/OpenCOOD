"""Freeze a trained current readout and test aligned complete-state local reuse."""
import json
from pathlib import Path
import torch
from opencood.models.sub_modules.int_current_readout import current_readout_from_mature
from opencood.models.sub_modules.int_short_readout import ShortReadoutState
from opencood.models.sub_modules.int_local_current_readout import LocalCurrentReadoutCache
from opencood.models.sub_modules.int_recurrent_deep import RecurrentDeepDetector
from opencood.tools.train_int_recurrent_deep import evaluate,cost_of
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.launch_int_memory import digest
from opencood.tools.run_int_ego import loader,to_device,write_json


def compare(a,cfg,train,val,baseline,out,costs,spatial_macs,deep_repair,checkpoint,source,expected):
    path=Path(a.current_readout_local);saved=torch.load(str(path),map_location='cpu');contract=saved['contract']
    assert contract['kind']=='current' and contract['frozen_deep_checkpoint']==checkpoint
    assert contract['source']=={k:source[k] for k in ('source','memory','readout','repair','basis','data')}
    trained=json.loads((path.parent/'report.json').read_text());assert trained['scans']==5794 and trained['labels']==1738
    calibration_path=Path(a.current_readout_local_calibration);cal=json.loads(calibration_path.read_text())
    assert cal['status']=='COMPLETED' and cal['split']=='train' and cal['scans']==512 and cal['fit']['status']=='FITTED'
    assert cal['identity']=={k:source[k] for k in cal['identity']}
    threshold=cal['fit']['threshold']
    repair,init=current_readout_from_mature(baseline.readout,contract['width'],False)
    repair.load_state_dict(saved['repair'],strict=True);repair=repair.cuda().eval()
    for p in repair.parameters():p.requires_grad_(False)
    modes={k:RecurrentDeepDetector(baseline,deep_repair,4) for k in ('dense_recurrent','current_dense','current_local','current_masked_dense')}
    modes['current_dense'].readout_cache=ShortReadoutState(baseline.readout,repair,baseline.bounds,3)
    for name,backend in (('current_local','selected'),('current_masked_dense','masked_dense')):
        modes[name].readout_cache=LocalCurrentReadoutCache(baseline.readout,repair,baseline.bounds,threshold,backend)
    policy=dict(checkpoint=str(path),sha256=digest(path),epoch=saved['epoch'],kind='current',
        calibration=dict(path=str(calibration_path),sha256=digest(calibration_path),threshold=threshold),
        selection='Transfer existing512-train calibrated threshold unchanged; compare actual current96 input to own retained reference; content dilation2, own confidence>=.1 dilation4, uncovered always selected. No new fit on validation.',
        execution='Whole completeY/reference/confidence aligned; first3x3 only core+halo1, second3x3 and final1x1 only core; first layer reads input halo2. All-core native dense fallback. Unselected alignedY retained.',
        refresh='Original mature full readout every3/.25s/reset, same as trained current model. D period4 unchanged.',
        controls='Independent mature D/full readout; trained dense current; selected local; dense-compute same masked policy. Real-weight same-input replay outside timing.',
        limitation='Transferred train calibration may be suboptimal for new readout; no threshold sweep or fabricated same-AP claim. Extra96 reference and packed weights included.')
    write_json(out/'local_current_contract.json',policy)
    if a.current_readout_profile:
        from opencood.tools.profile_int_current_readout import profile
        return profile(a,cfg,train,modes['current_local'],out,costs,spatial_macs)
    def replay(ego,previous,meta,pred,state,info):
        mode=modes['current_local'];cache=mode.readout_cache;mask=cache.last_mask.clone();cache.backend='masked_dense'
        try:other,other_state,_=mode.step(ego,previous,meta)
        finally:cache.backend='selected'
        assert torch.equal(mask,cache.last_mask)
        for k in pred:assert torch.allclose(pred[k],other[k],atol=1e-3,rtol=1e-3)
        assert torch.allclose(state['readout']['value'],other_state['readout']['value'],atol=1e-3,rtol=1e-3)
        assert torch.equal(state['readout']['reference'],other_state['readout']['reference'])
        return dict(prediction_max_error=max(float((pred[k]-other[k]).abs().max()) for k in pred),
            state_max_error=float((state['readout']['value']-other_state['readout']['value']).abs().max()))
    states={k:None for k in modes};records=[];partial=0
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            for name,mode in modes.items():
                previous=states[name];counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.repair,repair]))
                try:pred,states[name],info=mode.step(ego,previous,meta)
                finally:counter.close()
                actual=counter.total+info['spatial_extra_linear_macs']+info.get('readout_extra_linear_macs',0)
                assert actual==cost_of(costs,spatial_macs,info,True),(name,actual,cost_of(costs,spatial_macs,info,True))
                if name=='current_local':
                    records.append(replay(ego,previous,meta,pred,states[name],info))
                    partial+=int(not info['readout_refresh'] and info['readout_write_fraction']<1.)
    assert partial>0
    write_json(out/'local_current_engineering.json',dict(status='PASSED',frames=32,actual_sparse_frames=partial,actual_macs_verified=True,
        max_prediction_error=max(r['prediction_max_error'] for r in records),max_state_error=max(r['state_max_error'] for r in records)))
    audits=[];areas=[]
    def audit(name,ego,previous,meta,pred,state,info,index):
        if name=='current_local':
            areas.append(dict(scan=index+1,refresh=info['readout_refresh'],core=info['readout_write_fraction'],first_support=info['readout_first_support_fraction'],executed_macs=info['readout_executed_macs']))
            if index%200==0:
                audits.append(replay(ego,previous,meta,pred,state,info));write_json(out/'local_current_replays.json',audits)
    directory=out/'full';directory.mkdir()
    report=evaluate(a,baseline,None,cfg,val,directory,costs,spatial_macs,'local_current',variants=modes,
        expected_ap={'dense_recurrent':expected,'current_dense':trained['results']['short_readout']['ap70']},audit=audit,share_gpu=True,
        note='Frozen current checkpoint; completeY local reuse. Conv/functional MACsx2; full own trajectories and selection/warp/index costs included. Shared GPU allowed; affected whole latency null.')
    report['local_current_contract']=policy;report['same_input_replays']=len(audits)
    write_json(out/'areas.json',areas);write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED',scans=len(val)))
