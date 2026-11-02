"""Causal sparse readout backend with explicit persistent Conv accumulators."""
from torch import nn
from opencood.models.sub_modules.int_readout_events import ChangeEncodedReadout,ReadoutEventDetector
from opencood.models.sub_modules.int_channel_events import channel_event_delta


class SparseChangeEncodedReadout(ChangeEncodedReadout):
    def __init__(self,source,quanta):
        super().__init__(source,quanta);self.previous_accumulators=None;self.next_accumulators=None;self.execution=None

    def forward(self,x):
        self.next_accumulators=[];self.execution=[]
        result=super().forward(x);self.statistics['backend']='sparse_channel_event_delta';return result

    def convolve(self,layer,q,index):
        if self.refresh or self.previous is None:
            y=layer(q);stats=dict(refresh=True,executed_macs=q.shape[-2]*q.shape[-1]*layer.in_channels*layer.out_channels*layer.kernel_size[0]**2)
        else:
            if self.previous_accumulators is None:raise ValueError('Previous Conv accumulators required')
            delta,stats=channel_event_delta(layer,q-self.previous[index]);y=self.previous_accumulators[index]+delta;stats['refresh']=False
        self.next_accumulators.append(y.detach());self.execution.append(stats);return y


class SparseReadoutEventDetector(ReadoutEventDetector):
    def __init__(self,source,quanta):
        super().__init__(source,quanta);self.core.readout=SparseChangeEncodedReadout(source.core.readout,quanta);self.variant='protected_sparse_readout_events';self.eval()

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        readout=self.core.readout;readout.previous_accumulators=None if state is None else state['readout_accumulators']
        pred,new,info=super().step_spatial(spatial,state,meta,detach_state,requests)
        new['readout_accumulators']=readout.next_accumulators;info['cache_bytes']+=sum(v.numel()*v.element_size() for v in readout.next_accumulators);info['sparse_readout_execution']=readout.execution
        readout.previous_accumulators=None;readout.next_accumulators=None;return pred,new,info
