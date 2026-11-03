"""Frozen integer full-refresh encoders; floating task recurrence stays explicit.

Execution references belong to the last full-refresh encoder execution.
Ordinary frames skip both encoders without deleting their execution references.
This is an arithmetic engineering interface, not a trained SNN or AP result.
"""
from torch import nn
from opencood.models.sub_modules.int_fine_integer import FineIntegerConv
from opencood.models.sub_modules.int_fine_readout_integer import FineReadoutIntegerDetector
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason


ENCODER_PREFIXES = ('coarse_encoder.', 'deep_encoder.')


class RefreshEncoderIntegerDetector(FineReadoutIntegerDetector):
    def __init__(self, source, fine_calibration, readout_quanta, encoder_calibration,
                 encoder_mode, capture_audit=False):
        if encoder_mode not in ('dense', 'static', 'delta'):
            raise ValueError('Explicit fixed encoder execution mode required')
        super().__init__(source, fine_calibration, readout_quanta, 'hybrid', capture_audit)
        convs = {n: m for n, m in self.core.named_modules()
                 if isinstance(m, nn.Conv2d) and n.startswith(ENCODER_PREFIXES)}
        if len(convs) != 20 or set(encoder_calibration) != set(convs):
            raise ValueError('Exactly20 middle/deep encoder convolutions required')
        for name, conv in convs.items():
            parent, _, leaf = name.rpartition('.')
            replacement = FineIntegerConv(conv, encoder_calibration[name]['quanta'], encoder_mode)
            replacement.capture_audit = capture_audit
            self.core.get_submodule(parent)._modules[leaf] = replacement
        self.encoder_mode = encoder_mode
        self.variant = 'encoder_integer_' + encoder_mode
        self.eval()

    def encoder_leaves(self):
        return {n: m for n, m in self.core.named_modules()
                if n.startswith(ENCODER_PREFIXES) and isinstance(m, FineIntegerConv)}

    def step_spatial(self, spatial, state, meta, detach_state=True, requests=None):
        if not detach_state or requests is not None:
            raise ValueError('Frozen whole stream only')
        reason, _ = reset_reason(state, meta)
        previous = {} if reason else state['encoder_integer']
        leaves = self.encoder_leaves()
        if set(previous) - set(leaves):
            raise ValueError('Unknown encoder execution state')
        for name, leaf in leaves.items():
            leaf.previous = previous.get(name)
            leaf.next_state = None
            leaf.statistics = {}
            leaf.audit = None
        try:
            pred, new, info = super().step_spatial(spatial, state, meta, detach_state, requests)
            executed = {n: m for n, m in leaves.items() if m.statistics}
            expected = (set(leaves) if info['full_refresh'] else set())
            if set(executed) != expected:
                raise RuntimeError('Encoder execution schedule mismatch')
            retained = dict(previous) if self.encoder_mode == 'delta' else {}
            for name, leaf in executed.items():
                if leaf.next_state is not None:
                    retained[name] = leaf.next_state
            new['encoder_integer'] = retained
            info['encoder_integer'] = {n: dict(m.statistics) for n, m in executed.items()}
            capacity = sum(v.numel() * v.element_size() for pair in retained.values() for v in pair)
            info['encoder_cache_bytes'] = capacity
            info['cache_bytes'] += capacity
            info['audit_scratch_bytes'] += sum(v.numel() * v.element_size()
                                              for m in executed.values() if m.audit is not None for v in m.audit)
            return pred, new, info
        finally:
            for leaf in leaves.values():
                leaf.previous = None
                leaf.next_state = None

    def clear_audit(self):
        super().clear_audit()
        for leaf in self.encoder_leaves().values():
            leaf.audit = None
