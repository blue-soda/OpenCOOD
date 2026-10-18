"""Frame-local aligned latent accumulation, without intermediate quantization."""
from dataclasses import dataclass
import torch
from opencood.models.fuse_modules.e3dsnn_progressive import check_transform
from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume


@dataclass
class ContentState:
    owner: object
    contract: object
    pose: torch.Tensor
    scales: torch.Tensor
    source_counts: torch.Tensor
    aligned: torch.Tensor
    next_packet: int = 1
    closed: bool = False


class ContentAccumulator:
    def __init__(self, basis_sha256, family):
        if family not in ('count3', 'ann2bit'): raise ValueError('Unknown content family')
        self.basis_sha256, self.family = basis_sha256, family

    def begin(self, template, contract, pose, scales):
        check_transform(pose)
        if (contract.channel_basis != 'content32-' + self.basis_sha256 or
                template.shape != (1, 32) + contract.shape_zyx or
                scales.shape != (1, 32, 1, 1, 1) or
                not torch.isfinite(scales).all() or torch.any(scales <= 0) or
                scales.device != template.device or scales.dtype != template.dtype or
                pose.device != template.device):
            raise ValueError('Invalid content session geometry, basis or scales')
        if self.family == 'count3' and not torch.all(scales == 1):
            raise ValueError('Count3 requires unit scales')
        return ContentState(self, contract, pose.detach().clone(), scales.detach().clone(),
                            torch.zeros_like(template), torch.zeros_like(template))

    def receive(self, state, event, contract, pose, scales, packet_index):
        if state.owner is not self or state.closed or contract != state.contract:
            raise ValueError('Wrong session, frame or channel basis')
        if (type(packet_index) is not int or packet_index != state.next_packet or packet_index > 3 or
                not torch.equal(pose, state.pose) or not torch.equal(scales, state.scales)):
            raise ValueError('Duplicate, missing packet, changed pose or frame scales')
        if (event.shape != state.source_counts.shape or event.dtype != state.source_counts.dtype or
                event.device != state.source_counts.device or not torch.isfinite(event).all() or
                torch.any((event != 0) & (event != 1))):
            raise ValueError('Expected binary latent events on the source grid')
        # Unary levels are nested. Repeated coordinates across different levels
        # are intentional, but an event cannot first appear in a later level.
        if packet_index > 1 and torch.any((event > 0) & (state.source_counts < packet_index - 1)):
            raise ValueError('Non-nested event levels')
        aligned, coverage = warp_volume(event * scales, pose, contract.origin_xyz,
                                        contract.step_xyz, geometry_dtype=torch.float64)
        state.source_counts = state.source_counts + event
        state.aligned = state.aligned + aligned
        state.next_packet += 1
        return state.aligned, coverage

    def finish(self, state):
        if state.owner is not self or state.closed: raise ValueError('Wrong or closed session')
        state.closed = True
        return state.aligned
