"""Compose fixed fine quantization and fixed integer readout execution."""
from opencood.models.sub_modules.int_fine_integer import FineIntegerDetector
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadout
from opencood.models.sub_modules.int_implicit_integer_reference import ImplicitReferenceIntegerReadout
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason


class FineReadoutIntegerDetector(FineIntegerDetector):
    def __init__(self, source, fine_calibration, readout_quanta, mode, capture_audit=False):
        super().__init__(source, fine_calibration, mode, capture_audit)
        cls = IntegerReadout if mode == 'dense' else ImplicitReferenceIntegerReadout
        self.core.readout = cls(source.core.readout, readout_quanta, mode != 'dense')
        self.variant = 'fine_readout_integer_' + mode
        self.eval()

    def step_spatial(self, spatial, state, meta, detach_state=True, requests=None):
        reason, _ = reset_reason(state, meta)
        refresh = bool(reason)
        if not refresh:
            refresh = state['since_refresh'] + 1 >= self.period or (int(meta['timestamp_us']) - state['refresh_timestamp_us']) / 1e6 >= self.max_age - 1e-6
        readout = self.core.readout
        readout.previous = None if state is None else state['readout']
        readout.previous_accumulators = None if state is None else state['readout_accumulators']
        readout.refresh = refresh
        try:
            pred, new, info = super().step_spatial(spatial, state, meta, detach_state, requests)
            new['readout'] = readout.next_state
            new['readout_accumulators'] = readout.next_accumulators
            info['readout_events'] = readout.statistics
            info['cache_bytes'] += readout.statistics['sending_reference_bytes'] + readout.statistics['accumulator_bytes']
            return pred, new, info
        finally:
            readout.previous = None; readout.previous_accumulators = None
            readout.next_state = None; readout.next_accumulators = None
