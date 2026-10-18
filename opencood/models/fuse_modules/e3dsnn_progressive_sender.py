"""Sender-local balanced ordering with exact hard Count4 messages.

Only order is learned: the caller supplies a fixed selected voxel set. Soft
cumulative membership is a biased surrogate gradient, never a wire value.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class _HardMembership(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft, hard):
        return hard.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def balanced_partition(counts, scores, packets=3, temperature=1.):
    if counts.ndim!=5 or counts.shape[0]!=1 or scores.shape!=(1,1)+counts.shape[2:]:
        raise ValueError('One sender frame and one score per spatial voxel required')
    if packets<1 or packets>3 or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid registered partition parameters')
    if not torch.isfinite(counts).all() or torch.any((counts<0)|(counts>4)|(counts!=counts.floor())):
        raise ValueError('Exact Count4 input required')
    if not torch.isfinite(scores).all():raise ValueError('Nonfinite sender scores')
    active=torch.nonzero((counts[0]!=0).any(0).flatten()).flatten()
    n=active.numel()
    if not n:return [counts*0 for _ in range(packets)]
    values=scores.flatten()[active]
    # Stable ties retain ascending z/y/x address; no stochastic inference routing.
    # Torch 1.10 argsort lacks a stable keyword. CPU stable sorting is explicit
    # here; its synchronization/copy cost must be included in sender timing.
    order=torch.as_tensor(np.argsort(-values.detach().cpu().numpy(),kind='mergesort'),device=counts.device)
    cumulative=[];hard_cumulative=[]
    for k in range(1,packets):
        limit=(n*k+packets-1)//packets
        hard=torch.zeros_like(values);hard[order[:limit]]=1
        if limit==n:soft=torch.ones_like(values)
        else:
            threshold=((values[order[limit-1]]+values[order[limit]])*.5).detach()
            soft=torch.sigmoid((values-threshold)/temperature)
        cumulative.append(soft);hard_cumulative.append(hard)
    cumulative=[torch.zeros_like(values)]+cumulative+[torch.ones_like(values)]
    hard_cumulative=[torch.zeros_like(values)]+hard_cumulative+[torch.ones_like(values)]
    result=[]
    for k in range(packets):
        hard=hard_cumulative[k+1]-hard_cumulative[k]
        soft=cumulative[k+1]-cumulative[k]
        membership=_HardMembership.apply(soft,hard)
        mask=counts.new_zeros(counts.shape[2:].numel()).index_copy(0,active,membership)
        result.append(counts*mask.reshape(1,1,*counts.shape[2:]))
    return result


class ProgressiveSender(nn.Module):
    """Small ANN score head over Count4 and sender-local 3x3x3 mean context.

    This head is explicitly continuous, not an extra spiking backbone. It has
    no receiver, label, pose or detection-head inputs. Include its cost in reports.
    """
    def __init__(self,channels=128,hidden=16,temperature=1.):
        super().__init__();self.temperature=temperature
        self.score=nn.Sequential(nn.Conv3d(channels*2,hidden,1),nn.ReLU(),nn.Conv3d(hidden,1,1))

    def forward(self,counts,packets=3):
        normalized=counts/4.
        context=F.avg_pool3d(normalized,3,stride=1,padding=1,count_include_pad=False)
        scores=self.score(torch.cat((normalized,context),1))
        return balanced_partition(counts,scores,packets,self.temperature)
