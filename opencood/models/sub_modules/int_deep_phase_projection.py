"""Reuse disjoint upsampling phases under nearest-neighbor BEV alignment."""
import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.sub_modules.int_feature_memory import warp_bev


class DeepPhaseProjection:
    def __init__(self, block, indexed=False):
        if not isinstance(block, nn.Sequential) or len(block) != 3:
            raise ValueError('Expected a nonoverlapping projection, BatchNorm and ReLU')
        conv, norm, activation = block
        if (block.training or not isinstance(conv, nn.ConvTranspose2d)
                or not isinstance(norm, nn.BatchNorm2d) or not isinstance(activation, nn.ReLU)
                or conv.groups != 1 or conv.dilation != (1, 1) or any(conv.padding)
                or any(conv.output_padding) or conv.kernel_size != conv.stride
                or conv.stride[0] != conv.stride[1]):
            raise ValueError('Only square, disjoint, frozen upsampling tiles are supported')
        self.block = block
        self.stride = conv.stride[0]
        self.indexed = indexed

    def align_task(self, value, past_pose, current_pose, bounds):
        # Route using the same grid sampler and dtype as the floating task tensor.
        if value.dtype != torch.float32:
            raise ValueError('Indexed projection expects FP32 state')
        height, width = value.shape[-2:]
        if height*width > 2**24:
            raise ValueError('The cell indices must be representable in FP32')
        codes = torch.arange(1, height*width+1, dtype=value.dtype, device=value.device)
        sampled, valid = warp_bev(codes.view(1, 1, height, width), past_pose, current_pose, bounds)
        missing = sampled == 0
        indices = sampled.flatten().long().sub_(1).clamp_min_(0)
        aligned = value.flatten(2).index_select(2, indices).view_as(value)
        return aligned.masked_fill_(missing, 0.), valid, (indices, missing)

    def fresh(self, deep):
        projected = self.block(deep)
        return projected, F.pixel_unshuffle(projected, self.stride)

    def advance(self, deep, written, previous, past_pose, current_pose, bounds, route=None):
        """Update only written sites. The caller must write all newly exposed sites."""
        if route is not None:
            indices, missing = route
            # At an exact half-cell boundary grid_sample can return padding even
            # inside the geometric bounds. Materialize the projection of zero there.
            written = written | missing
        selected = written.flatten().nonzero().flatten()
        if selected.numel() == written.numel():
            return self.fresh(deep)
        if route is None:
            phases, _ = warp_bev(previous, past_pose, current_pose, bounds, mode='nearest')
            phases = phases.contiguous()
        else:
            phases = previous.flatten(2).index_select(2, indices).view_as(previous)
        if selected.numel():
            inputs = deep.flatten(2).index_select(2, selected).unsqueeze(2)
            _, updates = self.fresh(inputs)
            phases.flatten(2).index_copy_(2, selected, updates.flatten(2))
        return F.pixel_shuffle(phases, self.stride), phases
