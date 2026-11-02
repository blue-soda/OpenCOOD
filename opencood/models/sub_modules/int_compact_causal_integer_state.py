"""Lossless adaptive storage of integer execution state; arithmetic stays int64."""
import torch
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadout,IntegerReadoutDetector


def narrow_integer_state(value,reference=False):
    if value.dtype!=torch.int64:raise ValueError('Expected int64 arithmetic state')
    low,high=int(value.min()),int(value.max())
    if reference and -32768<=low and high<=32767:return value.to(torch.int16)
    if -(2**31)<=low and high<2**31:return value.to(torch.int32)
    return value


class CompactIntegerReadout(IntegerReadout):
    def forward(self,x):
        output=super().forward(x)
        self.next_state=[narrow_integer_state(v,True) for v in self.next_state]
        self.next_accumulators=[narrow_integer_state(v,False) for v in self.next_accumulators]
        self.statistics.update(sending_reference_bytes=sum(v.numel()*v.element_size() for v in self.next_state),accumulator_bytes=sum(v.numel()*v.element_size() for v in self.next_accumulators),
            reference_dtypes=[str(v.dtype) for v in self.next_state],accumulator_dtypes=[str(v.dtype) for v in self.next_accumulators],storage_policy='Lossless int16/int32 references and int32/int64 accumulators with per-frame range guard; MAC/update remains int64; conversion and range-reduction overhead not free')
        return output


class CompactIntegerReadoutDetector(IntegerReadoutDetector):
    def __init__(self,source,quanta,sparse=True):
        super().__init__(source,quanta,sparse);self.core.readout=CompactIntegerReadout(source.core.readout,quanta,sparse);self.variant='compact_integer_readout_sparse' if sparse else 'compact_integer_readout_dense';self.eval()
