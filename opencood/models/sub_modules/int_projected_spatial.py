"""Mature spatial blocks with a shared low-rank input-channel projection.

Transfer the projected-GRU idea: cheap dense 1x1 projection followed by a
selectable 3x3 update. Preserve the four original post-activation cache shapes,
BN, residuals and stem. Compressed ranks require dense accuracy recovery.
"""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_basis_spatial_state import SpatialBasisStage
from opencood.models.sub_modules.int_selected_conv import selected_conv3x3_rows


class ProjectedSpatialConv(nn.Module):
    def __init__(self, source, rank):
        super().__init__()
        if (not isinstance(source,nn.Conv2d) or source.kernel_size != (3,3) or
                source.stride != (1,1) or source.padding != (1,1) or
                source.dilation != (1,1) or source.groups != 1 or source.padding_mode != 'zeros'):
            raise ValueError('Expected unit-stride zero-padded 3x3 convolution')
        ci,co = source.in_channels,source.out_channels
        if not 1 <= rank <= min(ci,co*9):
            raise ValueError('Invalid input-channel projection rank')
        self.project = nn.Conv2d(ci,rank,1,bias=False).to(source.weight)
        self.mix = nn.Conv2d(rank,co,3,padding=1,bias=source.bias is not None).to(source.weight)
        # W[o,ky,kx,i] = U[o,ky,kx,r] S[r] Vh[r,i]. No nonlinearity
        # between projection and spatial convolution.
        matrix = source.weight.detach().cpu().double().permute(0,2,3,1).reshape(co*9,ci)
        u,s,vh = torch.linalg.svd(matrix,full_matrices=False)
        with torch.no_grad():
            self.project.weight.copy_(vh[:rank,:,None,None])
            self.mix.weight.copy_((u[:,:rank]*s[:rank]).reshape(co,3,3,rank).permute(0,3,1,2))
            if source.bias is not None: self.mix.bias.copy_(source.bias)
        self.retained_weight_energy = float(s[:rank].square().sum()/s.square().sum().clamp_min(1e-30))

    def forward(self,x):
        return self.mix(self.project(x))

    def selected(self,x,mask):
        if self.training or torch.is_grad_enabled():
            raise ValueError('Frozen selected inference required')
        return selected_conv3x3_rows(self.mix,self.project(x),mask)


class ProjectedSpatialStage(SpatialBasisStage):
    """Same four cached activations; train only the eight new kernel factors."""
    def __init__(self,source,rank):
        nn.Module.__init__(self)
        if len(source) != 3:
            raise ValueError('Expected three mature basic blocks')
        self.stem = copy.deepcopy(source[0])
        self.blocks = nn.ModuleList(copy.deepcopy(list(source)[1:]))
        self.rank = rank; self.retained_weight_energy = []
        self.spatial_input_projected = True
        for block in self.blocks:
            if block.downsample is not None:
                raise ValueError('Only identity residuals supported')
            for name in ('conv1','conv2'):
                layer = ProjectedSpatialConv(getattr(block,name),rank)
                self.retained_weight_energy.append(layer.retained_weight_energy)
                setattr(block,name,layer)
        for p in self.parameters(): p.requires_grad_(False)
        for block in self.blocks:
            for layer in (block.conv1,block.conv2):
                for p in layer.parameters(): p.requires_grad_(True)
        self.eval()

    def train(self,mode=True):
        super().train(mode)
        self.stem.eval()
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d): m.eval()
        return self
