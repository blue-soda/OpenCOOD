"""Paired full validation: fixed four-layer H cache + local readout cache."""
import copy
import json
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_local_readout import LocalReadoutCache
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.int_selection_audit import capture_step
from opencood.utils import eval_utils


def evaluate_readout(a,cfg,train,val,baseline,out,costs,spatial_macs):
    modes=dict(baseline=baseline)
    for name,execution in (('selected','selected'),('masked_dense','masked_dense'),('period1','selected')):
        mode=copy.copy(baseline)
        mode.readout_cache=LocalReadoutCache(baseline.readout,baseline.bounds,a.readout_threshold,
            execution=execution,period=1 if name=='period1' else a.readout_period)
        modes[name]=mode
    if a.readout_period==5:
        mode=copy.copy(baseline)
        mode.readout_cache=LocalReadoutCache(baseline.readout,baseline.bounds,a.readout_threshold,period=3)
        modes['period3']=mode
    feature_cache=modes['selected'].readout_cache
    cells=100*252
    readout_macs=cells*sum(c.weight.numel() for c in (feature_cache.first,feature_cache.project,feature_cache.second))
    assert 2*readout_macs==23533977600
    contract=json.loads((out/'contract.json').read_text())
    contract.update(run_kind='local_complete_readout_full_validation',readout_threshold=a.readout_threshold,
        comparison='Overrides prior generic mode descriptions: baseline is fixed four-layer planned-row single-H t.5/guard.1; selected adds local complete readout-output reuse; masked_dense shares this policy but calculates full readout before merging. For period5, period3 is the verified local readout control. All independently recurrent; no training/weight changes.',
        readout_cache=dict(input='actual current96 projection + corrected deep96 projection',reference='all96 input channels; retained outside output write mask',
            output='complete256-channel feature before unchanged dense heads',selector='4x4 tile relative RMS over all input channels; dilate changed outputs radius2, union previous OWN confidence>=.1 radius4 and uncovered',
            execution='first3x3+second1x1 on core+radius1; second3x3 only final core; complete current dependencies, zero padding; unselected aligned complete output reused',
            geometry='nearest metric SE2; output/reference/own confidence packed together',period=a.readout_period,max_full_age_s=feature_cache.max_full_age_us/1e6,
            period_scope='Only readout changes to period5/.45s when requested. Spatial H and deep D remain period3/.25s; FM gap reset unchanged; paired period3 readout included.',
            recurrence='independent FM,H,deep,readout per mode; next H and readout masks use own predictions, not fixed baseline masks',
            control='period1 matches baseline; independent masked_dense plus discarded same-input replay on any own-trajectory mask/output divergence',
            accounting='MACsx2 conv+functional linear only; extra256+96+1 state channels and fixed packed weights separate; model includes geometry/gather/scatter/copy; no hooks or replay in timing'))
    write_json(out/'contract.json',contract)

    def expected(name,info):
        value=costs[('dense',bool(info['local_partial']))]
        if not info['spatial_refresh']:value=value-spatial_macs+info['spatial_extra_linear_macs']
        if name!='baseline' and not info['readout_refresh'] and modes[name].readout_cache.execution=='selected':
            value=value-readout_macs+info['readout_extra_linear_macs']
        return value

    def replay(ego,previous,meta,pred,state,hmask,rmask):
        # This state/output is discarded; actual masked-dense trajectory is never replaced.
        other,other_state,_,other_mask=capture_step(modes['masked_dense'],ego,previous,meta)
        assert torch.equal(hmask,other_mask)
        assert torch.equal(rmask,modes['masked_dense'].readout_cache.last_mask)
        error=max(float((pred[k]-other[k]).abs().max()) for k in pred)
        assert all(torch.allclose(pred[k],other[k],atol=1e-3,rtol=1e-3) for k in pred),error
        assert torch.allclose(state['readout']['value'],other_state['readout']['value'],atol=1e-3,rtol=1e-3)
        assert torch.allclose(state['spatial']['maps'][0],other_state['spatial']['maps'][0],atol=1e-3,rtol=1e-3)
        return error

    states={k:None for k in modes}; maximum=period_error=0.;partial=0
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));predictions={}
            for name,mode in modes.items():
                previous=states[name]
                counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.repair]))
                try:pred,states[name],info,hmask=capture_step(mode,ego,previous,sample['meta'])
                finally:counter.close()
                actual=counter.total+info['spatial_extra_linear_macs']+info.get('readout_extra_linear_macs',0)
                assert actual==expected(name,info),(name,actual,expected(name,info))
                predictions[name]=pred
                if name=='selected':
                    partial+=not info['readout_refresh']
                    maximum=max(maximum,replay(ego,previous,sample['meta'],pred,states[name],hmask,mode.readout_cache.last_mask))
            for key in predictions['baseline']:
                left,right=predictions['baseline'][key],predictions['period1'][key]
                period_error=max(period_error,float((left-right).abs().max()))
                assert torch.equal(left,right),'Period1 must exactly recover fixed baseline'
    assert partial>0
    write_json(out/'readout_engineering.json',dict(status='PASSED',frames=32,partial_frames=partial,
        same_input_selected_vs_masked_max_error=maximum,period1_exact_max_error=period_error,
        actual_hook_plus_linear_macs_verified=True,readout_dense_macs=readout_macs,
        spatial_dense_macs=spatial_macs))
    del modes['period1']
    states={k:None for k in modes}
    metrics={k:dict(time=[],peak=[],macs=0,write=[],area=[],hwrite=[],state=[],full=0) for k in modes}
    stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    dataset,stream=loader(val,cfg,4,303);audits=[]
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'))
            names=list(modes);offset=index%len(names);names=names[offset:]+names[:offset]
            predictions={};masks={};previous=dict(states)
            for name in names:
                mode=modes[name];torch.cuda.synchronize();before=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                pred,states[name],info,hmask=capture_step(mode,ego,states[name],sample['meta'])
                rmask=mode.readout_cache.last_mask if mode.readout_cache else None
                torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tick)
                item=metrics[name]
                if index>=10:
                    item['time'].append(elapsed);item['peak'].append(max(0,torch.cuda.max_memory_allocated()-before))
                assert all(torch.isfinite(v).all() for v in pred.values())
                assert info.get('readout_full_age_s',0)<(mode.readout_cache.max_full_age_us/1e6 if mode.readout_cache else .25)
                item['macs']+=expected(name,info)
                item['write'].append(info.get('readout_write_fraction',1.))
                item['area'].append(info.get('readout_executed_fraction',1.))
                item['hwrite'].append(info['spatial_write_fraction'])
                item['full']+=int(info.get('readout_refresh',True))
                item['state'].append(info['state_bytes']+info['spatial_state_bytes']+info['anchor_bytes']+info.get('readout_state_bytes',0))
                predictions[name]=pred;masks[name]=(hmask,rmask)
                if sample['meta']['supervised']:
                    data=dict(ego=ego);boxes,scores=dataset.post.post_process(data,dict(ego=pred));gt=dataset.post.generate_gt_bbx(data)
                    for threshold in stats[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],threshold)
            same_masks=all(torch.equal(x,y) for x,y in zip(masks['selected'],masks['masked_dense']))
            close=all(torch.allclose(predictions['selected'][k],predictions['masked_dense'][k],atol=1e-3,rtol=1e-3) for k in predictions['selected'])
            if not same_masks or not close:
                error=replay(ego,previous['selected'],sample['meta'],predictions['selected'],states['selected'],*masks['selected'])
                audits.append(dict(scan=index+1,own_masks_equal=same_masks,own_predictions_close=close,same_input_max_error=error))
                write_json(out/'readout_recurrence_audits.json',dict(audits=audits))
                assert len(audits)<=128,'Frequent own-trajectory divergence requires review'
            if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='readout_validation',scans=index+1))
    assert index+1==len(val)
    results={}
    for name,item in metrics.items():
        results[name]=dict(ap70=eval_utils.calculate_ap(stats[name],.7)[0],ap50=eval_utils.calculate_ap(stats[name],.5)[0],
            mean_ms=float(np.mean(item['time'])),p95_ms=float(np.percentile(item['time'],95)),
            conv_linear_gflops=2*item['macs']/len(val)/1e9,mean_readout_write_fraction=float(np.mean(item['write'])),
            mean_readout_executed_fraction=float(np.mean(item['area'])),mean_spatial_write_fraction=float(np.mean(item['hwrite'])),
            mean_state_bytes=float(np.mean(item['state'])),max_incremental_cuda_peak_bytes=max(item['peak']),readout_full_frames=item['full'],
            extra_readout_packed_weight_bytes=0 if name=='baseline' else modes[name].readout_cache.packed_weight_bytes)
    assert abs(results['baseline']['ap70']-.7553159775482435)<1e-7,'Fixed four-layer reference drift'
    if 'period3' in results:
        assert abs(results['period3']['ap70']-.7552887126261554)<1e-7,'Period3 readout reference drift'
    write_json(out/'report.json',dict(status='COMPLETED',scans=len(val),labels=sum(r['supervised'] for r in val),results=results,
        readout_period=a.readout_period,readout_max_full_age_s=feature_cache.max_full_age_us/1e6,
        own_trajectory_audited_frames=len(audits),all_same_input_replays_passed=True,
        note='Weights fixed; spatial t.5 four-layer baseline held. Readout output cache changes own predictions and future H masks. No new training; no SNN attribution; FLOPs count conv/linear only.'))
    write_json(out/'status.json',dict(status='COMPLETED',scans=len(val)))
