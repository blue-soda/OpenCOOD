"""Local complete-D update with a dense-compute masked-policy correctness control."""
import copy
import json
from pathlib import Path
import torch
from opencood.models.sub_modules.int_recurrent_deep import RecurrentDeepDetector
from opencood.models.sub_modules.int_local_recurrent_deep import LocalRecurrentDeepDetector
from opencood.tools.int_artifact_paths import artifact_path
from opencood.tools.launch_int_memory import digest
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.train_int_recurrent_deep import cost_of,evaluate


def compare_local(a,cfg,train,val,baseline,out,costs,spatial_macs):
    reference=Path(a.recurrent_local_reference)/'report.json';report=json.loads(reference.read_text())
    assert report['status']=='COMPLETED' and report['scans']==5794 and report['labels']==1738
    checkpoint=report['frozen_contract']['checkpoints']['6'];path=artifact_path(checkpoint['path'])
    assert digest(path)==checkpoint['sha256'];saved=torch.load(path,map_location='cpu')
    source=json.loads((out/'contract.json').read_text());required={k:source[k] for k in ('source','memory','readout','repair','basis','data')}
    assert saved['contract']['period']==6 and saved['contract']['source']==required
    repair=copy.deepcopy(baseline.repair).eval();repair.load_state_dict(saved['repair'],strict=True)
    for p in repair.parameters():p.requires_grad_(False)
    period=a.recurrent_local_period
    expected=report['results']['trained6_period%d'%period]['ap70']
    if a.y_only_epochs:
        from opencood.tools.train_int_y_only import train_y_only
        return train_y_only(a,cfg,train,val,baseline,out,costs,spatial_macs,repair,checkpoint,source,expected)
    if a.current_readout_local:
        from opencood.tools.evaluate_int_local_current_readout import compare
        return compare(a,cfg,train,val,baseline,out,costs,spatial_macs,repair,checkpoint,source,expected)
    if a.short_readout_diagnostic:
        from opencood.tools.diagnose_int_short_readout import diagnose
        return diagnose(a,cfg,train,baseline,out,repair,checkpoint,source)
    if a.short_readout_epochs:
        from opencood.tools.train_int_short_readout import train_short_readout
        return train_short_readout(a,cfg,train,val,baseline,out,costs,spatial_macs,repair,checkpoint,source,expected)
    if a.recurrent_readout_calibration:
        from opencood.tools.evaluate_int_recurrent_readout import compare_composition
        return compare_composition(a,cfg,train,val,baseline,out,costs,spatial_macs,repair,checkpoint,reference,source,expected)
    modes=dict(dense_recurrent=RecurrentDeepDetector(baseline,repair,period),
        local_selected=LocalRecurrentDeepDetector(baseline,repair,period),
        local_masked_dense=LocalRecurrentDeepDetector(baseline,repair,period,'masked_dense'))
    contract=dict(checkpoint=checkpoint,reference_sha256=digest(reference),period=period,max_age_us=modes['dense_recurrent'].deep.max_age_us,
        selection='Actual own H recomputed mask dilated1 (repair output dependency), union uncovered D. H already includes own confidence protection. No extra descriptor or historical reference.',
        execution='First1x1 only on core+halo1, depthwise3x3 and last1x1 only core. All-core uses native dense repair. Unselected cells retain aligned COMPLETE D; each frame writes the complete mixed state.',
        controls='Same frozen weights/period, independent dense recurrence vs selected local policy vs dense-compute masked local policy. No training or threshold fit.',
        bounds='Short repair dense0.532224G per ordinary frame; total available saving limited to this small term. No claim of large total gain from local D. Index/gather/scatter overhead timed.',
        accounting='Conv hooks plus H and D functional MACs; no fabricated savings for masked_dense. State stays complete D plus existing H/FM, no additional persistent activation/reference.',
        validation='Baseline and dense recurrence AP must reproduce. Real-weight same-input local selected/masked replay32 train frames and every200 validation frames, discarded outside model timing.')
    write_json(out/'local_recurrent_contract.json',contract)
    def replay(ego,previous,meta,pred,state,info):
        mode=modes['local_selected'];mode.deep.execution='masked_dense'
        try:other,other_state,other_info=mode.step(ego,previous,meta)
        finally:mode.deep.execution='selected'
        assert torch.equal(info['spatial_update_mask'],other_info['spatial_update_mask'])
        errors=[float((pred[k]-other[k]).abs().max()) for k in pred]
        for k in pred:assert torch.allclose(pred[k],other[k],atol=1e-4,rtol=1e-4)
        left,right=state['deep']['value'],other_state['deep']['value']
        assert torch.allclose(left,right,atol=1e-4,rtol=1e-4)
        return dict(prediction_max_error=max(errors),deep_max_error=float((left-right).abs().max()))
    states={k:None for k in modes};records=[];partial=0
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'))
            for name,mode in modes.items():
                previous=states[name];counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.repair]))
                try:pred,states[name],info=mode.step(ego,previous,sample['meta'])
                finally:counter.close()
                actual=counter.total+info['spatial_extra_linear_macs']+info.get('deep_extra_linear_macs',0)
                assert actual==cost_of(costs,spatial_macs,info,True),(name,actual,cost_of(costs,spatial_macs,info,True))
                if name=='local_selected':
                    records.append(replay(ego,previous,sample['meta'],pred,states[name],info));partial+=int(info.get('deep_macs_saved',0)>0)
    assert partial>0,'Engineering must exercise actual skipped short computation'
    write_json(out/'local_recurrent_engineering.json',dict(status='PASSED',frames=32,actual_sparse_frames=partial,
        max_prediction_error=max(r['prediction_max_error'] for r in records),max_deep_error=max(r['deep_max_error'] for r in records),actual_macs_verified=True))
    audits=[]
    def audit(name,ego,previous,meta,pred,state,info,index):
        if name=='local_selected' and index%200==0:
            result=replay(ego,previous,meta,pred,state,info);result['scan']=index+1;audits.append(result)
            write_json(out/'local_recurrent_replays.json',audits)
    directory=out/'full';directory.mkdir()
    result=evaluate(a,baseline,None,cfg,val,directory,costs,spatial_macs,'local',variants=modes,expected_ap={'dense_recurrent':expected},audit=audit,
        note='Own independent histories; selected D reuse versus dense-compute masked-policy control. Same-input replays discarded outside model timing. Conv/functional MACsx2; index/gather/scatter/warp included in model time.')
    result['local_recurrent_contract']=contract;result['same_input_replays']=len(audits)
    write_json(out/'report.json',result);write_json(out/'status.json',dict(status='COMPLETED',scans=len(val)))
