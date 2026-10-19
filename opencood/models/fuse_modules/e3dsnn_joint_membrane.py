"""Local continuous state + aligned Count3 increments + subtractive residual.

The bounded Count4 readout Q is a state, not an always-positive emitted event.
R = U - Q includes inhibitory and saturation information. With retained R,
Q + R reconstructs U; adding the prior Q again elsewhere would double count.
The local-conditioned gain is fixed during a frame, so full-message integration
is additive. Learned ordering affects prefix detection, not the full current.
"""
from dataclasses import dataclass
import torch
from torch import nn
from opencood.models.fuse_modules.e3dsnn_progressive import check_transform, count_readout
from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume


@dataclass
class JointState:
    owner: object
    contract: object
    pose: torch.Tensor
    output: torch.Tensor
    residual: torch.Tensor
    gain: torch.Tensor
    seen: torch.Tensor
    next_packet: int = 1
    closed: bool = False


class JointMembraneReceiver(nn.Module):
    def __init__(self, channels=128, latent_channels=32, retain_residual=True,
                 use_local_membrane=True):
        super().__init__()
        self.channels, self.latent_channels = channels, latent_channels
        self.retain_residual = retain_residual
        self.use_local_membrane = use_local_membrane
        self.local_gate = nn.Conv3d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.local_gate.weight)
        self.current_scale = nn.Parameter(torch.tensor(.1))

    def begin(self, membrane, contract, pose, encoder_sha256, basis_sha256):
        if (contract.encoder_sha256 != encoder_sha256 or
                contract.channel_basis != 'joint-content32-' + basis_sha256):
            raise ValueError('Encoder or learned latent basis mismatch')
        if (membrane.shape != (1, self.channels) + contract.shape_zyx or
                not torch.isfinite(membrane).all()):
            raise ValueError('Invalid local membrane or grid')
        check_transform(pose)
        if pose.device != membrane.device:
            raise ValueError('Pose device mismatch')
        q = count_readout(membrane)
        u = membrane if self.use_local_membrane else q
        # Gate uses the same initial local state for every batch in this frame.
        gain = 2 * torch.sigmoid(self.local_gate(u)) * self.current_scale
        seen = torch.zeros((1, 1) + contract.shape_zyx, device=u.device, dtype=torch.bool)
        return JointState(self, contract, pose.detach().clone(), q, u - q, gain, seen)

    def receive(self, state, counts, contract, pose, packet_index, decode):
        if state.owner is not self or state.closed or state.contract != contract:
            raise ValueError('Wrong session, frame, timestamp or channel basis')
        if (type(packet_index) is not int or packet_index != state.next_packet or
                packet_index > 3 or not torch.equal(pose, state.pose)):
            raise ValueError('Duplicate, missing, out-of-order packet or changed pose')
        if (counts.shape != (1, self.latent_channels) + contract.shape_zyx or
                counts.device != state.output.device or counts.dtype != state.output.dtype or
                not torch.isfinite(counts).all() or
                torch.any((counts < 0) | (counts > 3) | (counts != counts.floor()))):
            raise ValueError('Expected Count3 integers on the registered source grid')
        active = (counts.detach() != 0).any(1, keepdim=True)
        if torch.any(active & state.seen):
            raise ValueError('Source voxel already delivered in an earlier packet')
        # Each learned batch owns whole latent rows. Decode only after 3D warp;
        # D explicitly translates latent channels into the local 128-D basis.
        aligned, coverage = warp_volume(counts, pose, contract.origin_xyz,
            contract.step_xyz, geometry_dtype=torch.float64)
        current = decode(aligned) * state.gain
        if current.shape != state.output.shape or not torch.isfinite(current).all():
            raise ValueError('Invalid latent-to-local current mapping')
        supported = (aligned.detach() != 0).any(1, keepdim=True) & (coverage > 0)
        u = state.output + state.residual + current
        q = count_readout(u)
        r = u - q if self.retain_residual else torch.zeros_like(u)
        state.output = torch.where(supported, q, state.output)
        state.residual = torch.where(supported, r, state.residual)
        state.seen = state.seen | active
        state.next_packet += 1
        return state.output

    def finish(self, state):
        if state.owner is not self or state.closed:
            raise ValueError('Wrong or closed receiver session')
        state.closed = True
        return state.output
