"""Identical quantized readout, initializing refresh accumulators with sparse events."""
from opencood.models.sub_modules.int_fine_readout_integer import FineReadoutIntegerDetector
from opencood.models.sub_modules.int_implicit_integer_reference import ImplicitReferenceIntegerReadout


class StaticRefreshReadoutDetector(FineReadoutIntegerDetector):
    def __init__(self,source,fine_calibration,readout_quanta,capture_audit=False):
        super().__init__(source,fine_calibration,readout_quanta,'hybrid',capture_audit)
        self.core.readout=ImplicitReferenceIntegerReadout(source.core.readout,readout_quanta,True,sparse_refresh=True)
        self.variant='fine_hybrid_static_refresh_readout'
        self.eval()
