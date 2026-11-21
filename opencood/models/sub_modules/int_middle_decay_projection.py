"""Frozen middle-state sender and exact linear receiver for refresh-age decay."""
import torch

from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection


class MiddleDecayProjection(DeepSpikeProjection):
    def __init__(self, block, predictor, execution):
        if predictor not in ('zero', 'rounded', 'factor', 'phase'):
            raise ValueError('Unknown decay predictor')
        quantum = block[0].weight.new_full((block[0].in_channels,), .25)
        super().__init__(block, quantum, execution)
        self.predictor = predictor
        denominators = (343,392,448,512) if predictor=='phase' else (1,8,64,512)
        self.scales = {d:self.scale / d for d in denominators}
        self.divisors = {d:self.scale.new_tensor(float(d)) for d in denominators}
        self.unit_tables = {}
        if execution == 'unit' and predictor in ('phase','factor'):
            self.unit_tables = {d:self.weight.int() * d for d in denominators[1:]}

    def begin(self, state, refresh):
        self.age = 0 if refresh is not None else state['since_refresh']+1
        self.previous = None if not self.age else state['coarse_projection']
        self.aligned = None
        self.leak = self.predictor != 'zero' and self.age in (1,2,3)
        phase = min(self.age,3)
        self.den = 8**phase if self.predictor in ('factor','phase') else 1
        if self.predictor == 'phase':
            self.den *= 7**(3-phase)

    def align_task(self, value, past_pose, current_pose, bounds):
        aligned,valid,route = super().align_task(value,past_pose,current_pose,bounds)
        indices,missing = route
        self.aligned = self.previous.flatten(2).index_select(2,indices).view_as(self.previous)
        self.aligned.masked_fill_(missing,0)
        return aligned,valid,route

    def __call__(self, value, written):
        target = value / self.quantum
        if not self.age:
            return value, target.round() * self.quantum
        before = self.aligned[:,:value.shape[1]]
        self.maintenance = None
        if self.predictor == 'phase':
            # A constant integer numerator now represents a decaying physical value.
            # Device tensor divisor preserves dyadic sender levels under FP32 division.
            if self.mode == 'dense':
                # Reduce the exact rational number to a dyadic before tie-to-even
                # rounding. Reciprocal multiplication by an odd divisor can perturb ties.
                previous_phase = min(self.age-1,3)
                odd = 7**(3-previous_phase)
                reduced = torch.div(before,odd,rounding_mode='trunc')
                assert torch.equal(reduced*odd,before)
                predicted = reduced.double() * ((.875 if self.leak else 1.)/8**previous_phase)
                self.events = (target.double()-predicted).round().long()
                self.numerator = ((predicted + self.events)*self.den).round().long()
            else:
                predicted = before.float() / self.divisors[self.den]
                self.events = (target-predicted).round().long()
                self.numerator = before + self.events*self.den
            return value, (self.numerator.float()/self.divisors[self.den])*.25
        if self.predictor == 'factor':
            prior = before * 7 if self.leak else before
            if self.mode == 'dense':
                old_den = self.den // 8 if self.leak else self.den
                prediction = before.double() / old_den * (.875 if self.leak else 1.)
                self.events = (target.double() - prediction).round().long()
                self.numerator = ((prediction + self.events) * self.den).long()
            else:
                self.events = (target - prior.float() / self.den).round().long()
                self.numerator = prior + self.events * self.den
        else:
            self.numerator = target.round().long()
            prior = (before.float() * .875).round().long() if self.leak else before
            if self.leak:
                self.maintenance = prior - before
            self.events = self.numerator - prior
        return value, self.numerator.float() * (.25 / self.den)

    def fresh(self, deep):
        if self.predictor != 'phase':
            return super().fresh(deep)
        level = self.levels(deep)
        numerator = level * self.den
        if int(numerator.abs().max()) * self.weight_bound >= 2**51:
            raise ValueError('Exact dense refresh bound exceeded')
        return self.finish(numerator,self.dense_linear(numerator),level,True,self.scales[self.den])

    def advance(self, deep, written, previous, past_pose, current_pose, bounds, route=None):
        if previous is not self.previous or route is None or self.aligned is None:
            raise ValueError('Receiver and task must share frame state and coordinate route')
        if int(self.numerator.abs().max()) * self.weight_bound >= 2**51:
            raise ValueError('Exact dense reference bound exceeded')
        linear = self.aligned[:,deep.shape[1]:]
        if self.mode == 'dense':
            linear = self.dense_linear(self.numerator)
        else:
            if self.leak and self.predictor == 'factor':
                linear.mul_(7)
            if self.maintenance is not None:
                self.event_linear(self.maintenance,linear)
            if self.mode == 'unit':
                self.event_linear(self.events,linear,unit_weight=self.unit_tables.get(self.den))
            else:
                amplitude = self.events * self.den if self.den != 1 else self.events
                self.event_linear(amplitude,linear)
        decoded, packed = self.finish(self.numerator,linear,self.events,False,self.scales[self.den])
        self.ledger.update(maintenance_units=0 if self.maintenance is None else int(self.maintenance.abs().sum()),
                           maintenance_addresses=0 if self.maintenance is None else int(self.maintenance.count_nonzero()),
                           scaled_levels=self.numerator.numel() if self.leak and self.predictor=='factor' else 0,
                           scaled_linear=linear.numel() if self.leak and self.predictor=='factor' else 0,
                           denominator=self.den)
        self.previous = self.aligned = self.events = self.maintenance = self.numerator = None
        return decoded, packed
