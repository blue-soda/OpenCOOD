"""Write resource-limited signed units directly to the detector's middle task membrane."""
import torch
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev


class NativeEventWriter:
    def __init__(self, drive, executor, budget, mode, verify):
        self.drive = drive
        self.executor = executor
        self.budget = budget
        self.mode = mode
        self.verify = verify
        self.adapter = drive.adapter
        self.charge = None

    def begin_frame(self, state, meta, shape, bounds, reason):
        if reason is not None:
            self.charge = None
        self.context = state, meta, shape, bounds

    def apply(self, fine, old, extra, output):
        cost = self.drive.apply(fine, old, extra, output)
        ids = extra.flatten().nonzero().flatten()
        state, meta, shape, bounds = self.context
        self.context = None
        older = old[0].flatten(1)[:, ids]
        output_flat = output[0].flatten(1)
        increment = output_flat[:, ids] - older
        prior = (older * 4).round()
        integrate = self.mode != 'discard'
        if integrate:
            if self.charge is None:
                aligned = torch.zeros_like(old)
            else:
                area = block_lidar_range(bounds, shape, old.shape[-2:], 4)
                aligned = warp_bev(self.charge, state['pose'], meta['pose'], area)[0]
            self.charge = aligned.masked_fill(~self.executor.skip, 0)
            previous = self.charge[0].flatten(1)[:, ids]
            if self.mode == 'misaligned':
                previous = previous.roll(len(ids)//2, 1)
            total = previous + increment
        else:
            total = increment
        valid = (total.abs() >= .125) & ((total > 0) | (prior > 0))
        eligible = valid.flatten().nonzero().flatten()
        grant = self.budget['addresses']
        count = min(len(eligible), grant)
        if count < len(eligible):
            scores = total.flatten()[eligible].abs()
            order = torch.argsort(scores, descending=True, stable=True)
            eligible = eligible[order[:count]]
        units = torch.zeros_like(total, memory_format=torch.contiguous_format)
        units.view(-1)[eligible] = total.flatten()[eligible].sign()
        # Convert to the detector's actual quarter-grid, including unserved cells.
        output_flat[:, ids] = (prior + units) * .25
        self.expected = units
        if integrate:
            remaining = total - units*.25
            self.charge[0].flatten(1)[:, ids] = remaining
        self.native = dict(grant=grant, demand=int(valid.sum()), sent=count, values=units.numel(),
                           charge_bytes=0 if not integrate else self.charge.numel()*self.charge.element_size())
        if self.verify and integrate:
            error = (previous.double()+increment.double()-remaining.double()-units.double()*.25).abs()
            self.native['balance_error'] = float(error.max()) if len(ids) else 0.
            assert torch.allclose(remaining+units*.25, total, atol=4e-7, rtol=1e-6)
        # Existing task drive already counted its two affine maps and residual adds.
        # Two power-of-two scales for the trained ternary correction, then difference,
        # prior scale, grid increment, grid decode, and (for carry) integrate/scale/reset.
        return cost + (6 + 3*int(integrate))*units.numel()

    def observe_events(self, events):
        ids = self.executor.refine_skip.extra.flatten().nonzero().flatten()
        actual = events[0].flatten(1)[:, ids]
        assert torch.equal(actual, self.expected)
        self.native['whole_addresses'] = int(events.count_nonzero())
        self.native['whole_units'] = int(events.abs().sum())
        self.expected = None
