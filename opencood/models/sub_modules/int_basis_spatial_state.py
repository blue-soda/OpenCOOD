"""Mature spatial kernels as cheap dense bases and selectable channel mixers.

Rank nine preserves each input channel's complete 3x3 spatial basis. Lower
ranks are approximations and require independent accuracy measurements. Four
post-activation maps, not residual/gate responses, form the reusable state.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class SpatialBasisConv(nn.Module):
    def __init__(self, source, rank):
        super().__init__()
        if (not isinstance(source, nn.Conv2d) or source.kernel_size != (3, 3)
                or source.stride != (1, 1) or source.padding != (1, 1)
                or source.dilation != (1, 1) or source.groups != 1
                or source.padding_mode != 'zeros' or not 1 <= rank <= 9):
            raise ValueError('Expected unit-stride zero-padded 3x3 and rank1..9')
        ci, co = source.in_channels, source.out_channels
        self.basis = nn.Conv2d(ci, ci*rank, 3, padding=1, groups=ci, bias=False).to(source.weight)
        self.mix = nn.Conv2d(ci*rank, co, 1, bias=source.bias is not None).to(source.weight)
        # Independent input-channel SVD; CPU float64 makes initialization
        # independent of GPU SVD implementation and keeps full-rank error small.
        weights = source.weight.detach().cpu().double().permute(1, 0, 2, 3).reshape(ci, co, 9)
        if min(co, 9) < rank:
            raise ValueError('Rank exceeds the per-input spatial matrix dimension')
        retained = total = 0.
        with torch.no_grad():
            for channel in range(ci):
                u, s, vh = torch.linalg.svd(weights[channel], full_matrices=False)
                self.basis.weight[channel*rank:(channel+1)*rank, 0].copy_(vh[:rank].reshape(rank, 3, 3))
                self.mix.weight[:, channel*rank:(channel+1)*rank, 0, 0].copy_(u[:, :rank]*s[:rank])
                retained += float(s[:rank].square().sum()); total += float(s.square().sum())
            if source.bias is not None:
                self.mix.bias.copy_(source.bias)
        self.retained_weight_energy = retained / max(total, 1e-30)

    def forward(self, x):
        return self.mix(self.basis(x))

    def selected(self, x, mask):
        """Dense cheap spatial bases, linear mixing only at requested cells.

        Returns selected output rows; functional MACs exclude basis convolution
        (counted by Conv2d hooks). Gather/scatter costs belong in wall time.
        """
        if self.training or torch.is_grad_enabled():
            raise ValueError('Frozen inference required')
        if x.shape[0] != 1 or mask.shape != (1, 1, *x.shape[-2:]) or mask.dtype != torch.bool:
            raise ValueError('Expected a matching single-frame boolean mask')
        selected = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        if not selected.numel():
            return x.new_empty((0, self.mix.out_channels)), selected, 0
        basis = self.basis(x)
        rows = basis.reshape(basis.shape[1], -1)[:, selected].t().contiguous()
        result = F.linear(rows, self.mix.weight[:, :, 0, 0], self.mix.bias)
        return result, selected, selected.numel()*self.mix.weight.numel()


def _selected_bn(rows, bn):
    # BN running statistics/affine only; no cross-cell dependency in eval mode.
    if bn.training or not bn.track_running_stats:
        raise ValueError('Fixed BN running statistics required')
    return F.batch_norm(rows, bn.running_mean, bn.running_var, bn.weight, bn.bias,
                        training=False, momentum=0., eps=bn.eps)


def _write_rows(old, indices, rows):
    result = old.reshape(old.shape[1], -1).clone()
    result[:, indices] = rows.t()
    return result.reshape_as(old)


class SpatialBasisStage(nn.Module):
    """Keep mature stem; replace four costly 3x3s by depthwise+pointwise pairs.

    Each partial frame reuses aligned post-ReLU conv1 and block-output maps
    for each of two blocks. Dense spatial bases see the mixed current/history
    preceding layer. There are no recomputed halo *outputs*. This is an
    explicit four-map state architecture, not the earlier single-map MLP.
    Alignment, selection and bounded refresh belong to the detector caller.
    """
    def __init__(self, source, rank=9):
        super().__init__()
        if len(source) != 3:
            raise ValueError('Expected three mature basic blocks')
        self.stem = copy.deepcopy(source[0])
        self.blocks = nn.ModuleList(copy.deepcopy(list(source)[1:]))
        self.rank = rank
        energy = []
        for block in self.blocks:
            if block.downsample is not None or block.conv1.in_channels != block.conv2.out_channels:
                raise ValueError('Only same-resolution identity residual blocks supported')
            for name in ('conv1', 'conv2'):
                conv = SpatialBasisConv(getattr(block, name), rank)
                energy.append(conv.retained_weight_energy)
                setattr(block, name, conv)
        self.retained_weight_energy = energy

    def observe(self, temporal):
        return self.stem(temporal)

    def dense_from_stem(self, current):
        maps = []
        for block in self.blocks:
            first = F.relu(block.bn1(block.conv1(current)))
            current = F.relu(block.bn2(block.conv2(first)) + current)
            maps.extend((first, current))
        return current, maps

    def forward(self, temporal):
        return self.dense_from_stem(self.observe(temporal))[0]

    def selected_from_stem(self, current, aligned_maps, mask, execution='selected'):
        """All four states use one externally chosen output mask.

        masked_dense is a numerically comparable reference, not a compute
        saving implementation. Inactive states remain exactly the supplied
        aligned whole activations, including residuals and nonlinearities.
        """
        if self.training or torch.is_grad_enabled():
            raise ValueError('Frozen inference required')
        if execution not in ('selected', 'masked_dense') or len(aligned_maps) != 4:
            raise ValueError('Expected four aligned maps and a supported execution')
        if current.shape[0] != 1 or mask.shape != (1, 1, *current.shape[-2:]) or mask.dtype != torch.bool:
            raise ValueError('Expected a matching single-frame boolean mask')
        if any(old.shape != current.shape or old.device != current.device or old.dtype != current.dtype for old in aligned_maps):
            raise ValueError('Aligned maps must match current stem')
        states = []; macs = 0
        for index, block in enumerate(self.blocks):
            old_first, old_out = aligned_maps[2*index:2*index+2]
            if execution == 'masked_dense':
                first = torch.where(mask, F.relu(block.bn1(block.conv1(current))), old_first)
                current = torch.where(mask, F.relu(block.bn2(block.conv2(first))+current), old_out)
            else:
                rows, ids, count = block.conv1.selected(current, mask); macs += count
                first = _write_rows(old_first, ids, F.relu(_selected_bn(rows, block.bn1))) if ids.numel() else old_first.clone()
                rows, ids, count = block.conv2.selected(first, mask); macs += count
                if ids.numel():
                    skip = current.reshape(current.shape[1], -1)[:, ids].t()
                    current = _write_rows(old_out, ids, F.relu(_selected_bn(rows, block.bn2)+skip))
                else:
                    current = old_out.clone()
            states.extend((first, current))
        return current, states, macs
