"""Frozen Z-cache history and reader counterfactuals on a fixed training prefix."""
import math
from pathlib import Path
import torch
from opencood.models.sub_modules.int_z_only import ZOnlyUpdater, ZOnlyDetector, make_reader
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_readout_diagnostic import readout_error
from opencood.models.sub_modules.int_anchor_residual import predict
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest


@torch.no_grad()
def observe_update(updater, evidence, own_old, teacher_old, valid):
    """Counterfactuals do not alter either history or the actual updater."""
    hidden=updater.features(torch.cat((evidence,own_old,valid.to(own_old.dtype)),1))
    candidate=updater.candidate(hidden)
    gate=.95*torch.sigmoid(updater.gate(hidden))*valid.to(own_old.dtype)
    recursive=updater(evidence,own_old,valid)
    assert torch.equal(recursive,gate*own_old+(1-gate)*candidate)
    return dict(recursive=recursive,teacher_history=updater(evidence,teacher_old,valid),
                own_warp_only=own_old,teacher_warp_only=teacher_old,candidate_only=candidate),gate


@torch.no_grad()
def observe_readers(mode, own, target):
    """Cross readers on identical Z without changing the recurrent trajectory."""
    return dict(fixed_on_own=mode.teacher_y(own),adapted_on_own=mode.reader(own),
                adapted_on_full=mode.reader(target))


def diagnose(a,cfg,train,baseline,out,source):
    path=Path(a.z_only_diagnostic);saved=torch.load(str(path),map_location='cpu')
    contract=saved['contract']
    assert contract['kind']=='z_only_spatial_v1' and contract['period']==3
    assert contract['source']=={k:source[k] for k in ('source','memory','readout','repair','basis','data')}
    updater=ZOnlyUpdater(width=contract['width'],context=contract.get('context','plain')).cuda().eval()
    updater.load_state_dict(saved['updater'],strict=True)
    for p in updater.parameters():p.requires_grad_(False)
    reader=make_reader(baseline.readout,False).cuda().eval()
    reader.load_state_dict(saved['reader'],strict=True)
    mode=ZOnlyDetector(baseline,updater,reader,3)
    write_json(out/'diagnostic_contract.json',dict(checkpoint=str(path),sha256=digest(path),epoch=saved['epoch'],
        width=contract['width'],reader=contract['reader'],frames=512,split='Fixed first512 training scans; no fitting or AP selection',
        comparison='Same E and signed Z: own recursion, previous full teacher Z history, own/teacher warp only, signed candidate only. Cross fixed/adapted readers on the SAME own Z and complete current Z.',
        causality='Previous teacher Z is separate diagnostic buffer; never returned as student state. Current fixed teacher only a target. Full/ordinary inference and supervised replay must match exactly on every scan.',
        metrics='Pooled feature squared error/target energy and elements by depth1/2, all/teacher-confidence>=.1 foreground/uncovered. Gate sum and cell count, not frame-weighted means. Reader Y errors use the independent fixed full-Z reader as target; crossed reader outputs are never written into state.',
        limits='Teacher-history and candidate-only interventions are off-policy diagnostics, not attainable AP or latency/FLOPs results. Training-prefix evidence alone cannot establish generalization or causal architectural superiority.'))
    state=None;previous_target=None;totals={};counts={};gate_totals={};refresh=0
    _,stream=loader(train[:512],cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];previous=state
            pred,state,info=mode.step(ego,previous,meta,supervise=True)
            other,replayed,_=mode.step(ego,previous,meta)
            assert all(torch.equal(pred[k],other[k]) for k in pred)
            assert torch.equal(state['z']['value'],replayed['z']['value'])
            target=info['z_target']
            if info['z_refresh']:refresh+=1
            else:
                assert previous_target is not None
                assert previous_target['stamp']==previous['z']['stamp'] and previous_target['scene']==previous['z']['scene']
                snapshot=previous['z']['value'].clone()
                evidence,_,_=mode.front(ego,previous,meta)
                own_old,valid=warp_bev(previous['z']['value'],previous['z']['pose'],meta['pose'],mode.cache.bounds)
                teacher_old,teacher_valid=warp_bev(previous_target['value'],previous_target['pose'],meta['pose'],mode.cache.bounds)
                assert torch.equal(valid,teacher_valid)
                observations,gate=observe_update(updater,evidence,own_old,teacher_old,valid)
                assert torch.equal(observations['recursive'],state['z']['value'])
                assert torch.equal(snapshot,previous['z']['value'])
                depth=str(info['z_depth']);counts[depth]=counts.get(depth,0)+1
                if depth=='1':
                    assert torch.equal(own_old,teacher_old)
                    assert torch.equal(observations['recursive'],observations['teacher_history'])
                foreground=predict(mode.model,info['y_target'])['psm'].sigmoid().amax(1,keepdim=True)>=.1
                for region,mask in (('all',None),('foreground',foreground),('uncovered',~valid)):
                    for name,value in observations.items():
                        key='z/'+depth+'/'+name+'/'+region
                        error=readout_error(value,target,mask)
                        total=totals.setdefault(key,{k:0. for k in error})
                        for k,v in error.items():total[k]+=v
                    if region=='all':
                        reader_values=observe_readers(mode,observations['recursive'],target)
                        assert torch.equal(reader_values['adapted_on_own'],info['y_student'])
                        if contract['reader']=='frozen':
                            assert torch.equal(reader_values['adapted_on_own'],reader_values['fixed_on_own'])
                            assert torch.equal(reader_values['adapted_on_full'],info['y_target'])
                    for name,value in reader_values.items():
                        key='y/'+depth+'/'+name+'/'+region
                        error=readout_error(value,info['y_target'],mask)
                        total=totals.setdefault(key,{k:0. for k in error})
                        for k,v in error.items():total[k]+=v
                    selected=torch.ones_like(valid) if mask is None else mask
                    total=gate_totals.setdefault(depth+'/'+region,dict(gate_sum=0.,cells=0))
                    total['gate_sum']+=float(gate.masked_select(selected).sum());total['cells']+=int(selected.sum())
            previous_target=dict(value=target,pose=meta['pose'],stamp=int(meta['timestamp_us']),scene=meta['segment'])
            if (index+1)%100==0:write_json(out/'status.json',dict(status='RUNNING',phase='diagnostic',scans=index+1))
    assert index+1==512 and set(counts)=={'1','2'}
    metrics={}
    for key,total in totals.items():
        metrics[key]=dict(total,rmse=math.sqrt(total['squared_error']/total['elements']) if total['elements'] else None,
            relative_rmse=math.sqrt(total['squared_error']/total['target_energy']) if total['target_energy'] else None)
    for total in gate_totals.values():total['mean_gate']=total['gate_sum']/total['cells'] if total['cells'] else None
    write_json(out/'report.json',dict(status='COMPLETED',scans=512,refresh_frames=refresh,ordinary_by_depth=counts,
        actual_trajectory_exact=True,teacher_never_written=True,reader_swap_same_z=True,metrics=metrics,gates=gate_totals,
        note='Frozen training-prefix feature diagnosis only; no AP, runtime, savings or generalization claim.'))
    write_json(out/'status.json',dict(status='COMPLETED',scans=512))
