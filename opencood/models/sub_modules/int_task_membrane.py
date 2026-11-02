"""Inference-only task-potential and signed sigma-delta communication interface.

This is not a complete SNN conversion: the candidate stays dense and continuous.
Potential replaces the recurrent task state; communicated value is additional
execution state. Alignment and downstream accumulators are caller costs.
"""
import copy,math
import torch
from torch import nn


class TaskPotentialEvents(nn.Module):
    def __init__(self,candidate,quantum=.25,tau=.2):
        super().__init__()
        if quantum<=0 or tau<=0:raise ValueError('Positive quantum and tau required')
        self.candidate=copy.deepcopy(candidate);self.quantum=float(quantum);self.tau=float(tau)

    def forward(self,current,old_potential,old_sent,valid,dt):
        if torch.is_grad_enabled():raise ValueError('Inference interface; no surrogate training yet')
        if not 0<=dt<=.25:raise ValueError('Reset on noncausal time or gaps')
        old=torch.where(valid,old_potential,torch.zeros_like(old_potential))
        sent=torch.where(valid,old_sent,torch.zeros_like(old_sent))
        candidate=self.candidate(current,old,valid);z=1-math.exp(-dt/self.tau)
        potential=torch.where(valid,old+z*(candidate-old),candidate)
        counts=torch.trunc((potential-sent)/self.quantum)
        communicated=sent+self.quantum*counts
        assert bool(torch.isfinite(communicated).all())
        error=(potential-communicated).abs()
        stats=dict(active_elements=int(torch.count_nonzero(counts)),elements=counts.numel(),
            signed_unit_events=int(counts.abs().double().sum()),positive_events=int(counts.clamp(min=0).double().sum()),
            negative_events=int((-counts).clamp(min=0).double().sum()),max_abs_count=int(counts.abs().max()),
            max_representation_error=float(error.max()),state_bytes=2*potential.numel()*potential.element_size(),
            dense_pointwise_elements=potential.numel())
        return potential,communicated,counts,stats
