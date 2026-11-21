"""Write resource-limited signed units directly to the detector's middle task membrane."""
import numpy as np
import torch
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev


class SignedProjectionAccumulator:
    """Add a prepacked signed weight on ±1 inputs and multiply only larger changes."""
    def __init__(self, projector, verify):
        self.positive = projector.weight
        self.negative = (-projector.weight.int()).short()
        self.original = projector.event_linear
        self.verify = verify
        self.stats = None

    def __call__(self, changes, linear):
        gold = linear.clone() if self.verify else None
        grid = changes.flatten(2)[0]
        channels, sites = grid.nonzero(as_tuple=True)
        values = grid[channels, sites]
        target = linear.flatten(2)[0]
        fanout = self.positive.shape[1]
        chunk = max(1, 1048576//fanout)
        for key, mask in [('positive',values==1),('negative',values==-1),('multiple',values.abs()>1)]:
            src, dst, vals = channels[mask], sites[mask], values[mask]
            table = self.negative if key=='negative' else self.positive
            for j in range(0,len(src),chunk):
                ix = slice(j,j+chunk)
                weights = table[src[ix]].long()
                if key == 'multiple':
                    weights = weights * vals[ix,None]
                target.index_add_(1,dst[ix],weights.T)
        one = int((values.abs()==1).sum())
        self.stats = dict(ac=one*fanout, mac=(len(values)-one)*fanout, fanout=fanout,
                          extra_weight_bytes=self.negative.numel()*self.negative.element_size())
        if self.verify:
            self.original(changes,gold)
            assert torch.equal(gold,linear)


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
            ranked = np.argsort(-scores.cpu().numpy(), kind='stable')
            order = torch.as_tensor(ranked[:count], device=eligible.device)
            eligible = eligible[order[:count]]
        units = torch.zeros_like(total, memory_format=torch.contiguous_format)
        chosen = total.flatten()[eligible]
        if self.mode == 'amplitude':
            # At the exact half-quantum tie the admitted address emits at least one unit.
            amount = chosen.sign() * (chosen.abs()*4).round().clamp_min(1)
            released = torch.maximum(amount, -prior.flatten()[eligible])
            assert bool((released != 0).all())
        else:
            released = chosen.sign()
        units.view(-1)[eligible] = released
        # Convert to the detector's actual quarter-grid, including unserved cells.
        output_flat[:, ids] = (prior + units) * .25
        self.expected = units
        if integrate:
            remaining = total - units*.25
            self.charge[0].flatten(1)[:, ids] = remaining
        self.native = dict(grant=grant, demand=int(valid.sum()), sent=count, values=units.numel(),
                           max_magnitude=float(released.abs().max()) if count else 0.,
                           sent_units=int(released.abs().sum()),
                           charge_bytes=0 if not integrate else self.charge.numel()*self.charge.element_size())
        if self.verify and integrate:
            error = (previous.double()+increment.double()-remaining.double()-units.double()*.25).abs()
            self.native['balance_error'] = float(error.max()) if len(ids) else 0.
            assert torch.allclose(remaining+units*.25, total, atol=4e-7, rtol=1e-6)
        # Existing task drive already counted its two affine maps and residual adds.
        # Two power-of-two scales for the trained ternary correction, then difference,
        # prior scale, grid increment, grid decode, and (for carry) integrate/scale/reset.
        return cost + (6 + 3*int(integrate))*units.numel() + (2*count if self.mode=='amplitude' else 0)

    def observe_events(self, events):
        ids = self.executor.refine_skip.extra.flatten().nonzero().flatten()
        actual = events[0].flatten(1)[:, ids]
        assert torch.equal(actual, self.expected)
        self.native['whole_addresses'] = int(events.count_nonzero())
        self.native['whole_units'] = int(events.abs().sum())
        if getattr(self, 'capture_addresses', False):
            self.addresses = (events!=0).sum(1,keepdim=True)*self.executor.refine_skip.extra
        self.expected = None
