"""Task residual communication primitives, with explicit recurrent state.

Competitive cap is per INTERNAL update, not a transport packet quota. Finite
unroll, cap and stable magnitude ordering are declared design parameters.
Discrete active-set selection uses a biased straight-through estimator.
"""
import numpy as np
import torch
from torch import nn


def signed_if(v):
    hard = torch.sign(v) * (v.abs() >= 1).to(v.dtype)
    # Nonzero surrogate on both sides of the firing threshold. A clipped
    # identity would kill all selected-event gradients for |v|>1.
    soft = torch.tanh(v)
    return soft + (hard-soft).detach()


def ann_quant(v):
    soft = v.clamp(-3,3)
    return soft + (soft.round()-soft).detach()


def select_symbols(symbols, priority, limit, step):
    """Stable priority/address ties; candidate cap is an ablatable compute bound."""
    hard = symbols.detach().cpu().numpy().ravel()
    score = priority.detach().cpu().double().numpy().ravel()
    ids = np.flatnonzero(hard != 0)
    ids = ids[np.lexsort((ids, -score[ids]))[:limit]]
    events = [(step,int(i),int(hard[i])) for i in ids]
    return events


def masked_symbols(symbols, events):
    mask = torch.zeros_like(symbols).reshape(-1)
    if events:
        ids=torch.as_tensor([e[1] for e in events],device=symbols.device,dtype=torch.long)
        mask[ids]=1
    return symbols * mask.reshape_as(symbols)


class ResidualSender(nn.Module):
    def __init__(self, channels=128, latent=32):
        super().__init__()
        self.project=nn.Conv3d(channels,latent,1,bias=False)
        self.context=nn.Conv3d(latent,latent,3,padding=1,groups=latent,bias=False)
        self.decode=nn.Conv3d(latent,channels,1,bias=False)
        nn.init.zeros_(self.context.weight)
        nn.init.orthogonal_(self.project.weight.flatten(1))
        with torch.no_grad():self.decode.weight.copy_(self.project.weight.transpose(0,1))

    def drive(self,residual):
        x=self.project(residual)
        return x+self.context(x)

    def step(self,source,reconstruction,potential,family):
        drive=self.drive(source-reconstruction)
        if family=='residual_snn':
            value=potential+drive
            return signed_if(value),value
        if family=='residual_ann':
            return ann_quant(drive),drive
        raise ValueError('Unknown progressive sender family')


class ResidualReceiver(nn.Module):
    """Local+remote context changes drive a subtractive-reset hidden state.

    Replays logical steps in order; never resets at transport boundaries. The
    final continuous correction is produced from accumulated signed outputs.
    Analog state/ordinary convolution remain: this is a hybrid SNN design.
    """
    def __init__(self,channels=128,latent=32):
        super().__init__()
        self.input=nn.Conv3d(2*channels+1,latent,1,bias=False)
        self.spatial=nn.Conv3d(latent,latent,3,padding=1,groups=latent,bias=False)
        self.output=nn.Conv3d(latent,channels,1,bias=False)
        nn.init.zeros_(self.spatial.weight)
        nn.init.normal_(self.output.weight,std=.01)

    def context(self,local,remote,support):
        x=self.input(torch.cat((local,remote,support),1))
        return x+self.spatial(x)

    def begin(self,local):
        remote=torch.zeros_like(local);support=local[:,:1]*0
        context=self.context(local,remote,support)
        return dict(previous=context,potential=torch.zeros_like(context),total=torch.zeros_like(context))

    def receive(self,state,local,remote,support,family):
        context=self.context(local,remote,support)
        delta=context-state['previous'];state['previous']=context
        if family=='residual_snn':
            potential=state['potential']+delta
            spike=signed_if(potential)
            state['potential']=potential-spike
            state['total']=state['total']+spike
            hidden=state['total']
        elif family=='residual_ann':
            state['potential']=state['potential']+delta
            hidden=torch.tanh(state['potential'])
        else:raise ValueError('Unknown receiver family')
        return self.output(hidden)*support
