"""Frame-local progressive fusion prototype with explicit spatial/basis contracts.

No cross-frame cache; positive wire counts can cause signed local corrections.
With unit gate, keeping residual is algebraically equivalent to summing currents
before quantization. That is a correctness control, not an accuracy claim.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume


@dataclass(frozen=True)
class MessageContract:
    receiver_frame: str
    sender_frame: str
    sender_id: str
    encoder_sha256: str
    channel_basis: str
    shape_zyx: tuple
    origin_xyz: tuple
    step_xyz: tuple
    timestamps_us: tuple

    def __post_init__(self):
        if not all(isinstance(x, str) and x for x in (self.receiver_frame,self.sender_frame,
                self.sender_id,self.channel_basis)):
            raise ValueError('Explicit frame, sender and channel-basis identities required')
        if len(self.encoder_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.encoder_sha256):
            raise ValueError('Expected encoder SHA256, not architecture name')
        if len(self.shape_zyx)!=3 or any(not isinstance(n,int) or n<=0 for n in self.shape_zyx):
            raise ValueError('Invalid zyx grid shape')
        if len(self.origin_xyz)!=3 or len(self.step_xyz)!=3 or not all(math.isfinite(x) for x in self.origin_xyz+self.step_xyz):
            raise ValueError('Invalid xyz geometry')
        if any(x<=0 for x in self.step_xyz):raise ValueError('Positive grid steps required')
        if len(self.timestamps_us)!=2 or not all(isinstance(x,int) for x in self.timestamps_us):
            raise ValueError('Preserve both actual sensor timestamps; do not assert synchronization')


def check_transform(matrix):
    if matrix.shape!=(1,4,4) or not torch.isfinite(matrix).all():
        raise ValueError('Expected finite single-frame SE(3) matrix')
    rotation=matrix[0,:3,:3];eye=torch.eye(3,device=matrix.device,dtype=matrix.dtype)
    if not torch.allclose(rotation.T@rotation,eye,atol=1e-5,rtol=1e-5) or not torch.allclose(torch.det(rotation),matrix.new_tensor(1.),atol=1e-5,rtol=1e-5):
        raise ValueError('Expected rigid proper rotation')
    if not torch.equal(matrix[0,3],matrix.new_tensor([0,0,0,1])):
        raise ValueError('Invalid homogeneous transform')


def count_readout(value):
    """Count4 forward, bounded surrogate derivative; independent of legacy class."""
    continuous=value.clamp(0,4)
    rounded=torch.floor(continuous+.5)
    return continuous+(rounded-continuous).detach()


@dataclass
class FrameState:
    owner: object
    contract: MessageContract
    transform: torch.Tensor
    output: torch.Tensor
    residual: torch.Tensor
    next_packet: int = 0
    closed: bool = False


class ProgressiveMembraneFusion(nn.Module):
    """Shared pointwise basis mapping, then gated membrane integration on ego grid.

    Current prototype intentionally accepts batch=1 and one sender. Caller must
    open a new state per frame. Different source/ego grid definitions are rejected.
    """
    def __init__(self,channels=128,retain_residual=True,adaptive_gate=False):
        super().__init__();self.channels=channels
        self.retain_residual=retain_residual;self.adaptive_gate=adaptive_gate
        self.basis=nn.Conv3d(channels,channels,1,bias=False)
        self.gate=nn.Conv3d(3*channels,channels,1,bias=False)
        with torch.no_grad():
            self.basis.weight.copy_(torch.eye(channels).reshape(channels,channels,1,1,1))
            self.gate.weight.zero_()
        self.gate.weight.requires_grad_(adaptive_gate)

    def _shape(self,x,contract):
        if x.shape!=(1,self.channels)+contract.shape_zyx or not torch.isfinite(x).all():
            raise ValueError('Wrong frame batch, channels, grid shape or nonfinite features')

    def begin(self,ego_precount,contract,source_to_ego,encoder_sha256,channel_basis):
        if contract.encoder_sha256!=encoder_sha256 or contract.channel_basis!=channel_basis:
            raise ValueError('Sender/receiver feature bases do not match')
        self._shape(ego_precount,contract);check_transform(source_to_ego)
        if source_to_ego.device!=ego_precount.device:raise ValueError('Transform device mismatch')
        u=self.basis(ego_precount);q=count_readout(u)
        return FrameState(self,contract,source_to_ego.detach().clone(),q,
                          u-q if self.retain_residual else torch.zeros_like(u))

    def receive(self,state,counts,contract,source_to_ego,packet_index):
        if state.owner is not self or state.closed:raise ValueError('Wrong or closed receiver session')
        if contract!=state.contract:raise ValueError('Frame, timestamp, grid or channel-basis mismatch')
        if not torch.equal(source_to_ego,state.transform):raise ValueError('Pose changed within one frame')
        if not isinstance(packet_index,int) or packet_index!=state.next_packet:
            raise ValueError('Duplicate, missing or out-of-order packet')
        self._shape(counts,contract)
        if counts.device!=state.output.device or counts.dtype!=state.output.dtype:
            raise ValueError('Message dtype/device mismatch')
        if torch.any((counts<0)|(counts>4)|(counts!=torch.floor(counts))):
            raise ValueError('Wire message must contain Count4 integer values')
        # Align complete 3D positions before basis mapping; never align after Z folding.
        aligned,coverage=warp_volume(counts,source_to_ego,contract.origin_xyz,contract.step_xyz)
        supported=(aligned!=0).any(1,keepdim=True)&(coverage>0)
        current=self.basis(aligned)
        if self.adaptive_gate:
            current=current*(2*torch.sigmoid(self.gate(torch.cat((state.output,state.residual,current),1))))
        candidate=state.output+state.residual+current
        q=count_readout(candidate)
        residual=candidate-q if self.retain_residual else torch.zeros_like(candidate)
        # No received support means exact identity; packet omission is not evidence of free space.
        q=torch.where(supported,q,state.output)
        residual=torch.where(supported,residual,state.residual)
        correction=q-state.output  # may be negative locally; not a signed wire count
        state.output,state.residual=q,residual;state.next_packet+=1
        return correction

    def finish(self,state):
        if state.owner is not self or state.closed:raise ValueError('Wrong or closed receiver session')
        state.closed=True
        return state.output


def split_counts(counts,packets=3):
    """Fixed spatial-row partition for correctness audits, NOT a learned sender."""
    if counts.ndim!=5 or counts.shape[0]!=1 or packets<1:raise ValueError('Single frame required')
    spatial=torch.arange(counts.shape[2]*counts.shape[3]*counts.shape[4],device=counts.device).reshape(1,1,*counts.shape[2:])
    return [counts*(spatial%packets==i).to(counts.dtype) for i in range(packets)]
