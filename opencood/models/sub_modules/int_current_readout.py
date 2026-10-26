"""Mature-initialized current readout with optional bounded complete-state reuse."""
import copy
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


class CurrentReadoutRepair(nn.Module):
    def __init__(self,channels=256,width=64,current_channels=96,gated=True):
        super().__init__()
        self.channels,self.current_channels,self.gated=channels,current_channels,gated
        # No activation between the second spatial factor and output factor:
        # together they approximate one mature convolution before its ReLU.
        self.net=nn.Sequential(nn.Conv2d(current_channels,width,3,padding=1),nn.ReLU(),
            nn.Conv2d(width,width,3,padding=1,bias=False),nn.Conv2d(width,channels,1),nn.ReLU())
        if gated:
            self.gate=nn.Conv2d(current_channels+3,1,1)
            nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)

    def forward(self,packed):
        current=packed[:,:self.current_channels]
        fresh=self.net(current)
        if not self.gated:return fresh
        old=packed[:,self.current_channels:self.current_channels+self.channels]
        valid=packed[:,-1:]
        evidence=torch.cat((current,old.mean(1,keepdim=True),
            (fresh-old).abs().mean(1,keepdim=True),valid),1)
        weight=.95*self.gate(evidence).sigmoid()*valid
        return weight*old+(1-weight)*fresh


def current_readout_from_mature(readout,width=64,gated=True):
    """Prune first channels and SVD-factor the fused second spatial map.

    Frozen BN is folded into convolutions. Keeping every channel with full rank
    reproduces the mature function; reduced width is explicitly approximate.
    """
    first=fuse_conv_bn_eval(copy.deepcopy(readout.spatial).cpu().eval(),copy.deepcopy(readout.tail[0]).cpu().eval())
    project=readout.tail[2][0];second=readout.tail[2][1]
    assert project.kernel_size==(1,1) and project.bias is None
    second=fuse_conv_bn_eval(copy.deepcopy(second).cpu().eval(),copy.deepcopy(readout.tail[3]).cpu().eval())
    assert first.kernel_size==second.kernel_size==(3,3)
    channels=first.out_channels
    assert channels==second.out_channels and 0<width<=channels
    combined=torch.einsum('orkl,ri->oikl',second.weight.detach().double(),project.weight.detach().cpu().double()[:,:,0,0])
    score=first.weight.detach().double().flatten(1).norm(dim=1)*combined.permute(1,0,2,3).flatten(1).norm(dim=1)
    selected=score.topk(width,sorted=True).indices
    matrix=combined[:,selected].reshape(channels,-1)
    u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
    result=CurrentReadoutRepair(channels,width,first.in_channels,gated).eval()
    with torch.no_grad():
        result.net[0].weight.copy_(first.weight[selected]);result.net[0].bias.copy_(first.bias[selected])
        root=s[:width].sqrt()
        result.net[2].weight.copy_((root[:,None]*vh[:width]).reshape(width,width,3,3).float())
        result.net[3].weight.copy_((u[:,:width]*root).reshape(channels,width,1,1).float())
        result.net[3].bias.copy_(second.bias)
    return result,dict(first_channels=selected.tolist(),second_svd_energy=float(s[:width].square().sum()/s.square().sum()),
        initialization='Fold mature frozen BN; select first channels by input/output weight norm product; rank-factor second spatial map before ReLU. No training/validation samples used.',
        history='Complete output state only; gated old coefficient in[0,.95], zero outside coverage. Not a proof of whole-network contraction.')
