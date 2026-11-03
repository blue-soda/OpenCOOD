"""Fixed fine/readout quantization with differentiable ordinary task updates.

Fine is frozen and evaluated without a graph. Readout uses the existing exact
integer forward and STE input derivative. Only ordinary task/light modules
may be opened by the learning tool; no second task memory is introduced.
"""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_fine_integer import FineIntegerEncoder
from opencood.models.sub_modules.int_causal_readout_qat import CausalReadoutQAT
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector


STATE_QAT_PREFIXES=('core.coarse_update.','core.deep_update.','core.coarse_light.','core.deep_light.')


def load_state_update_targets(model, targets):
    """Validate every target before changing any parameter; keep frozen weights."""
    expected={n:p for n,p in model.named_parameters() if n.startswith(STATE_QAT_PREFIXES)}
    if not expected or set(targets)!=set(expected):raise ValueError('Exact ordinary-state target keys required')
    for n,p in expected.items():
        value=targets[n]
        if not isinstance(value,torch.Tensor) or value.shape!=p.shape or value.dtype!=p.dtype:
            raise ValueError('Target shape/dtype mismatch: '+n)
        if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite target: '+n)
    with torch.no_grad():
        for n,p in expected.items():p.copy_(targets[n])
    return sum(p.numel() for p in expected.values())


def causal_update_indices(rows, refresh_reasons, window_ends):
    if len(rows)!=len(refresh_reasons):raise ValueError('Schedule length mismatch')
    ordinary=eligible=False;result=[]
    for i,(row,reason) in enumerate(zip(rows,refresh_reasons)):
        if reason in ('initial','scene','gap','explicit'):ordinary=False
        ordinary|=not bool(reason)
        eligible|=bool(row['supervised']) and ordinary
        if i in window_ends:
            if eligible:result.append(i)
            ordinary=eligible=False
    return result


class FrozenFineObservation(FineIntegerEncoder):
    def forward(self,x):
        if any(p.requires_grad for p in self.parameters()):raise ValueError('Fine parameters must be frozen')
        with torch.no_grad():return super().forward(x)


class JointStateQATDetector(ProtectedMultiscaleRefreshDetector):
    def __init__(self,source,fine_calibration,readout_quanta,enabled):
        nn.Module.__init__(self)
        if type(source) is not ProtectedMultiscaleRefreshDetector:raise ValueError('Protected source required')
        self.core=copy.deepcopy(source.core);self.pillar_vfe,self.scatter=source.pillar_vfe,source.scatter
        for key in ('bounds','period','max_age','light_observations','memory_policy','spatial_only'):setattr(self,key,copy.deepcopy(getattr(source,key)))
        self.enabled=bool(enabled)
        if enabled:self.core.observation=FrozenFineObservation(source.core.observation,fine_calibration,'dense')
        self.core.readout=CausalReadoutQAT(source.core.readout,readout_quanta,enabled)
        self.variant='joint_state_qat' if enabled else 'joint_state_ann'
        for p in self.parameters():p.requires_grad_(False)
        self.eval()

    def fine_reference_flops(self):
        return sum(m.statistics['actual_fp64_flops'] for m in self.core.observation.leaves().values()) if self.enabled else 0

    def open_ordinary_updates(self):
        for name,p in self.named_parameters():p.requires_grad_(name.startswith(STATE_QAT_PREFIXES))

