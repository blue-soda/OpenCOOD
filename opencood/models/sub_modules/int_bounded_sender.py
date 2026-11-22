"""A bounded-error sender feeding the same full-precision temporal convolution receiver."""
import torch

from opencood.models.sub_modules.int_shallow_anchor import AnchoredObservationConv


class BoundedObservationConv(AnchoredObservationConv):
    tolerance = 1 / 16

    def __init__(self, original, history, signed=True, allow_unit=True):
        super().__init__(original, signed=signed, signed_rebuild=signed)
        self.history = history
        self.allow_unit = allow_unit
        self.previous = None

    def begin(self, name, reason, meta, input_shape, bounds, verify):
        super().begin(name, reason, meta, input_shape, bounds, verify)
        if reason is not None:
            self.previous = None

    def forward(self, x):
        refresh = self.anchor_state is None
        emitted = {}
        if refresh:
            received = x
        else:
            positive = x != 0
            if self.history:
                prior = self.previous
                difference = x - prior
                hold = (difference.abs() <= self.tolerance) & (positive | (prior == 0))
                if self.allow_unit:
                    candidate = prior + torch.where(difference > 0, self.quantum, -self.quantum)
                    spike = ((x - candidate).abs() <= self.tolerance) & (candidate >= 0)
                    spike &= positive | (candidate == 0)
            else:
                prior = 0.
                candidate = self.quantum
                hold = x <= self.tolerance
                spike = (x - candidate).abs() <= self.tolerance
            # The two admissible intervals are disjoint at these fixed scales.
            correction = torch.where(spike, candidate, x) if self.allow_unit else x
            received = torch.where(hold, prior, correction)
            held, unit = int(hold.sum()), int(spike.sum()) if self.allow_unit else 0
            emitted = dict(hold=held, unit=unit, exact=x.numel()-held-unit)
        error = (received - x).abs()
        if self.verify:
            assert bool((error <= self.tolerance).all())
            assert bool(((x != 0) | (received == 0)).all()) and float(received.min()) >= 0
        output = super().forward(received)
        if self.history:
            # At refresh the exact observation is already owned by the receiver.
            self.previous = self.anchor_state[0] if refresh else received
            if not refresh:
                self.record['state_bytes'] += received.numel() * received.element_size()
        self.record.update(emitted, error_l1=float(error.mean()), sender_max_error=float(error.max()))
        return output
