"""Compose one latent task-state membrane with exact integer readout execution."""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_subthreshold_membrane import SubthresholdMembraneDetector
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadout
from opencood.models.sub_modules.int_compact_causal_integer_state import CompactIntegerReadout


class JointMembraneReadoutDetector(SubthresholdMembraneDetector):
    def __init__(self,source,quanta,sparse=False):
        nn.Module.__init__(self)
        if type(source) is not SubthresholdMembraneDetector:raise ValueError('Single task-membrane source required')
        self.core=copy.deepcopy(source.core);self.pillar_vfe,self.scatter=source.pillar_vfe,source.scatter
        for key in ('bounds','memory_policy','spatial_only','period','max_age','light_observations','membrane_mode','quantum'):setattr(self,key,copy.deepcopy(getattr(source,key)))
        readout=CompactIntegerReadout if sparse else IntegerReadout
        self.core.readout=readout(source.core.readout,quanta,sparse);self.variant='joint_membrane_integer_sparse' if sparse else 'joint_membrane_integer_dense';self.eval()

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        if torch.is_grad_enabled() or not detach_state:raise ValueError('Frozen joint execution only')
        reason,_=reset_reason(state,meta);refresh=bool(reason)
        if not refresh:refresh=state['since_refresh']+1>=self.period or (int(meta['timestamp_us'])-state['refresh_timestamp_us'])/1e6>=self.max_age-1e-6
        readout=self.core.readout;readout.refresh=refresh
        readout.previous=None if state is None else state['readout']
        readout.previous_accumulators=None if state is None else state['readout_accumulators']
        pred,new,info=super().step_spatial(spatial,state,meta,detach_state,requests)
        new['readout']=readout.next_state;new['readout_accumulators']=readout.next_accumulators
        info['readout_events']=readout.statistics
        info['cache_bytes']+=readout.statistics['sending_reference_bytes']+readout.statistics['accumulator_bytes']
        readout.previous=readout.next_state=readout.previous_accumulators=readout.next_accumulators=None
        return pred,new,info
