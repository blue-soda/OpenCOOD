"""Same combined quantization with sparse readout refresh initialization."""
from opencood.models.sub_modules.int_combined_integer import CombinedIntegerDetector


class ComposedStaticIntegerDetector(CombinedIntegerDetector):
    def __init__(self,source,fine_calibration,readout_quanta,candidate_calibration,
                 encoder_calibration,capture_audit=False):
        super().__init__(source,fine_calibration,readout_quanta,candidate_calibration,
                         encoder_calibration,'static','static',capture_audit)
        self.core.readout.sparse_refresh=True
        self.variant='combined_static_integer_with_static_readout_refresh'
        self.eval()
