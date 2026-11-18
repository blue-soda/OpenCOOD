"""Reuse disjoint upsampling phases under nearest-neighbor BEV alignment."""
from torch import nn
from torch.nn import functional as F

from opencood.models.sub_modules.int_feature_memory import warp_bev


class DeepPhaseProjection:
    def __init__(self, block):
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

    def fresh(self, deep):
        projected = self.block(deep)
        return projected, F.pixel_unshuffle(projected, self.stride)

    def advance(self, deep, written, previous, past_pose, current_pose, bounds):
        """Update only written sites. The caller must write all newly exposed sites."""
        selected = written.flatten().nonzero().flatten()
        if selected.numel() == written.numel():
            return self.fresh(deep)
        phases, _ = warp_bev(previous, past_pose, current_pose, bounds, mode='nearest')
        phases = phases.contiguous()
        if selected.numel():
            inputs = deep.flatten(2).index_select(2, selected).unsqueeze(2)
            _, updates = self.fresh(inputs)
            phases.flatten(2).index_copy_(2, selected, updates.flatten(2))
        return F.pixel_shuffle(phases, self.stride), phases
