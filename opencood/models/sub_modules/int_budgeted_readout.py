"""Cost-bounded causal readout selection from cheap current input only."""
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_cache_repair import expand_tiles
from opencood.models.sub_modules.int_local_readout import LocalReadoutCache


def executed_macs(mask,w1,w2):
    support=F.max_pool2d(mask.float(),3,1,1)>0
    return int(support.sum())*w1+int(mask.sum())*w2


def budgeted_ranking(scores,mandatory,budget,w1,w2):
    """Largest nested score prefix within actual halo-aware row MAC budget.

    This is not globally optimal subset selection. Costs include the final
    mask's radius1 intermediates, not just its number of selected output cells.
    """
    if scores.ndim!=4 or scores.shape[:2]!=(1,1) or not torch.isfinite(scores).all():
        raise ValueError('Single finite score grid required')
    if mandatory.dtype!=torch.bool or mandatory.shape[:2]!=(1,1):
        raise ValueError('Single boolean mandatory mask required')
    if min(w1,w2)<=0 or executed_macs(mandatory,w1,w2)>budget:
        raise ValueError('Mandatory costs must fit budget')
    order=scores.flatten().argsort(descending=True)
    def candidate(k):
        chosen=torch.zeros_like(scores,dtype=torch.bool)
        chosen.flatten()[order[:k]]=True
        core=F.max_pool2d(expand_tiles(chosen,4,mandatory.shape[-2:]).float(),5,1,2)>0
        mask=core|mandatory
        return mask,executed_macs(mask,w1,w2)
    lo,hi=0,order.numel();best=mandatory.clone();cost=executed_macs(best,w1,w2)
    while lo<hi:
        mid=(lo+hi+1)//2;mask,value=candidate(mid)
        if value<=budget:lo=mid;best=mask;cost=value
        else:hi=mid-1
    assert cost<=budget and bool((best|~mandatory).all())
    return best,dict(tiles=lo,macs=cost,budget=budget,budget_fraction=cost/max(budget,1))


class BudgetedReadoutCache(LocalReadoutCache):
    def __init__(self,readout,bounds,threshold,execution='selected',period=3,budget_scale=1.,ranking='absolute'):
        super().__init__(readout,bounds,threshold,execution,period)
        if budget_scale not in (.8,1.) or period not in (1,3) or threshold!=.5:
            raise ValueError('Bounded absolute ranking requires threshold.5 and budget.8/1, period1/3')
        if ranking not in ('absolute','relative'):raise ValueError('Unknown input-error ranking')
        self.ranking=ranking
        self.budget_scale=budget_scale;self.last_selector_stats=None

    def select_mask(self,current,reference,valid,confidence):
        relative=super().select_mask(current,reference,valid,confidence)
        mandatory=(F.max_pool2d((~valid).float(),5,1,2)>0)|(F.max_pool2d((confidence>=.1).float(),9,1,4)>0)
        w1=self.first.weight.numel()+self.project.weight.numel();w2=self.second.weight.numel()
        reference_macs=executed_macs(relative,w1,w2)
        mandatory_macs=executed_macs(mandatory,w1,w2)
        requested=int(reference_macs*self.budget_scale);budget=max(requested,mandatory_macs)
        absolute=F.avg_pool2d((current-reference).square().mean(1,keepdim=True),4,4,ceil_mode=True,count_include_pad=False)
        scores=absolute
        if self.ranking=='relative':
            pool=lambda x:F.avg_pool2d(x,4,4,ceil_mode=True,count_include_pad=False)
            energy=torch.maximum(pool(current.square().mean(1,keepdim=True)),pool(reference.square().mean(1,keepdim=True)))
            scores=absolute/energy.clamp_min(1e-6)
        mask,stats=budgeted_ranking(scores,mandatory,budget,w1,w2)
        self.last_selector_stats=dict(readout_relative_reference_macs=reference_macs,
            readout_selector_macs=stats['macs'],readout_selector_budget=budget,
            readout_budget_protection_override=mandatory_macs>requested)
        return mask

    def step(self,current,state,meta,reset=False):
        self.last_selector_stats=None
        output,new,info=super().step(current,state,meta,reset)
        if self.last_selector_stats is not None:info.update(self.last_selector_stats)
        return output,new,info


