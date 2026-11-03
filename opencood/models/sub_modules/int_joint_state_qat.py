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

