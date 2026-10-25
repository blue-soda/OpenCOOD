"""Train-only feasibility of selecting before the mature spatial stem."""
import json
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_cache_repair import expand_tiles
from opencood.models.sub_modules.int_early_cache import early_descriptor
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.int_selection_audit import capture_step
from opencood.tools.run_int_ego import loader,to_device,write_json


def dependency_cost(mask,weights):
    """Exact output-cell arithmetic for hypothetical selected stem + four U.

    Stem stride2 first conv executes on M+5 at the H grid, second and skip
    on M+4. U projection/mix supports are4/3/2/1 and3/2/1/0.
    This counts dense-length dot products including zero-padding, not latency.
    """
    counts=[int(mask.sum())];support=mask
    for _ in range(5):
        support=F.max_pool2d(support.float(),3,1,1)>0;counts.append(int(support.sum()))
    u=sum(counts[4-i]*p+counts[3-i]*m for i,(p,m) in enumerate(weights['u']))
    stem=counts[5]*weights['stem'][0]+counts[4]*sum(weights['stem'][1:])
    return u,stem


def score_mask(scores,mandatory,threshold):
    return expand_tiles(scores>threshold,4,mandatory.shape[-2:])|mandatory


def fit_stem_threshold(scores,mandatory,weights,target):
    def cost(t):
        return sum(dependency_cost(score_mask(s,m,t),weights)[0]
                   for s,m in zip(scores.split(16),mandatory.split(16)))
    lo=0.;hi=float(scores.max());floor=cost(hi)
    if floor>target:raise ValueError('Required protection exceeds comparison U budget')
    for _ in range(24):
        mid=(lo+hi)/2
        if cost(mid)<=target:hi=mid
        else:lo=mid
    return hi,cost(hi)


def audit_stem(a,cfg,train,mode,out):
    stage=mode.spatial.stage;stem=stage.stem
    assert stem.conv1.stride==(2,2) and stem.conv2.stride==(1,1)
    assert stem.conv1.kernel_size==stem.conv2.kernel_size==(3,3)
    assert stem.downsample[0].kernel_size==(1,1) and stem.downsample[0].stride==(2,2)
    weights=dict(stem=[stem.conv1.weight.numel(),stem.conv2.weight.numel(),stem.downsample[0].weight.numel()],
        u=[(c.project.weight.numel(),c.mix.weight.numel()) for b in stage.blocks for c in (b.conv1,b.conv2)])
    contract=json.loads((out/'contract.json').read_text())
    contract.update(run_kind='training_pre_stem_selection_feasibility',
        stem_audit='Frozen four-layer H baseline, no readout cache. Train first512 only. Compare pre-stem GRU group16/pool2 descriptor to reference retained at canonical H updates. Fit one scalar to canonical aggregate U MACs. Counterfactual masks/teacher H never enter history. Additional dense H is discarded. Potential stem costs include both convs and skip, all dependencies. No selected stem backend, closed-loop AP or deployed latency claim.')
    write_json(out/'contract.json',contract)
    _,stream=loader(train[:a.stem_audit_frames],cfg,0,303)
    state=None;cheap_reference=None;records=[];full=0;exact=0.;dense_cost=None
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));captured={};original=stage.observe
            def observed(temporal):
                captured['cheap']=early_descriptor(temporal,2)
                current=original(temporal);captured['current']=current
                return current
            stage.observe=observed;previous=state
            try:pred,new,info,mask=capture_step(mode,ego,previous,sample['meta'])
            finally:stage.observe=original
            cheap=captured['cheap'];current=captured['current']
            assert cheap.shape[-2:]==mask.shape[-2:]
            if dense_cost is None:
                dense_cost=dependency_cost(torch.ones_like(mask),weights)
                assert 2*sum(dense_cost)==36333158400
            if info['spatial_refresh']:
                cheap_reference=cheap;full+=1
            else:
                old=previous['spatial'];packed=torch.cat((old['maps'][0],cheap_reference,old['confidence']),1)
                aligned,valid=warp_bev(packed,old['pose'],sample['meta']['pose'],mode.bounds)
                historical=aligned[:,:128];reference=aligned[:,128:144]
                pool=lambda x:F.avg_pool2d(x,4,4,ceil_mode=True,count_include_pad=False)
                error=pool((cheap-reference).square().mean(1,keepdim=True))
                energy=torch.maximum(pool(cheap.square().mean(1,keepdim=True)),pool(reference.square().mean(1,keepdim=True)))
                scores=(error/energy.clamp_min(1e-6)).sqrt()
                mandatory=(~valid)|(F.max_pool2d((aligned[:,-1:]>=.1).float(),9,1,4)>0)
                dense,_=stage.dense_from_stem(current)
                assert torch.allclose(new['spatial']['maps'][0].masked_select(mask),dense.masked_select(mask),atol=1e-3,rtol=1e-3)
                costs=dependency_cost(mask,weights)
                assert costs[0]==info['spatial_extra_linear_macs']
                records.append(dict(scores=scores,mandatory=mandatory,mask=mask,
                    error=(historical-dense).square().mean(1,keepdim=True),energy=dense.square().mean(1,keepdim=True),cost=costs))
                cheap_reference=torch.where(mask,cheap,reference)
            if index<8:
                other,other_state,_=mode.step(ego,previous,sample['meta'])
                for key in pred:
                    exact=max(exact,float((pred[key]-other[key]).abs().max()));assert torch.equal(pred[key],other[key])
                assert torch.equal(new['spatial']['maps'][0],other_state['spatial']['maps'][0])
                assert torch.equal(new['spatial']['reference'],other_state['spatial']['reference'])
            state=new
            if (index+1)%100==0:write_json(out/'status.json',dict(status='RUNNING',phase='stem_feasibility',scans=index+1))
        assert index+1==a.stem_audit_frames
        target=sum(r['cost'][0] for r in records)
        threshold,matched=fit_stem_threshold(torch.cat([r['scores'] for r in records]),
            torch.cat([r['mandatory'] for r in records]),weights,target)
        result={}
        for name in ('canonical_dense_stem','canonical_selected_stem_oracle','cheap_gru_descriptor'):
            macs=full*sum(dense_cost);error=energy=cells=xor=0.
            for r in records:
                mask=r['mask'] if name!='cheap_gru_descriptor' else score_mask(r['scores'],r['mandatory'],threshold)
                u,stem_cost=dependency_cost(mask,weights)
                macs+=u+(dense_cost[1] if name=='canonical_dense_stem' else stem_cost)
                error+=float((r['error']*~mask).sum());energy+=float((r['energy']*~mask).sum())
                cells+=int(mask.sum());xor+=int((mask^r['mask']).sum())
            result[name]=dict(stem_plus_u_gflops=2*macs/(index+1)/1e9,
                retained_error=error,retained_relative_rmse=(error/max(energy,1e-12))**.5,
                partial_write_fraction=cells/(len(records)*mask.numel()),partial_mask_xor=xor/(len(records)*mask.numel()))
        for item in result.values():item['retained_error_vs_canonical']=item['retained_error']/max(result['canonical_dense_stem']['retained_error'],1e-12)
    write_json(out/'stem_audit.json',dict(status='COMPLETED',split='train',scans=index+1,full_frames=full,
        partial_frames=len(records),threshold=threshold,matched_u_cost_ratio=matched/target,
        exact_replay_max_error=exact,results=result,weights=weights,
        limitation='Potential arithmetic, not executed savings. Canonical selected-stem mask needs dense E and is an oracle cost reference. Cheap descriptor calibrated on frozen canonical histories; its own recurrent H/reference/guard may differ. No AP, validation tuning, or latency claim.'))
    write_json(out/'status.json',dict(status='COMPLETED',phase='stem_feasibility',scans=index+1))
