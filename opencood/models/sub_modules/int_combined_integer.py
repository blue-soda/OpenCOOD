"""Compose frozen candidate and refresh encoder integer execution.

References have distinct schedules: fine candidates run every frame, middle/deep
candidates run ordinary frames, encoders run full refreshes. Each retains its
last executed input until scene/gap reset. Nonlinearities and task gates stay
floating point. Joint trajectories must be audited; marginal costs cannot add.
"""
from torch import nn
from opencood.models.sub_modules.int_fine_integer import FineIntegerConv
from opencood.models.sub_modules.int_refresh_encoder_integer import RefreshEncoderIntegerDetector
from opencood.models.sub_modules.int_candidate_integer import CANDIDATE_PREFIXES
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason


class CombinedIntegerDetector(RefreshEncoderIntegerDetector):
    def __init__(self, source, fine_calibration, readout_quanta, candidate_calibration,
                 encoder_calibration, candidate_mode, encoder_mode, capture_audit=False):
        if candidate_mode not in ('dense', 'static', 'delta'):
            raise ValueError('Explicit fixed candidate execution mode required')
        super().__init__(source, fine_calibration, readout_quanta, encoder_calibration, encoder_mode, capture_audit)
        convs = {n: m for n, m in self.core.named_modules()
                 if isinstance(m, nn.Conv2d) and n.startswith(CANDIDATE_PREFIXES)}
        if len(convs) != 15 or set(candidate_calibration) != set(convs):
            raise ValueError('Exactly three five-convolution candidate updates required')
        for name, conv in convs.items():
            parent, _, leaf = name.rpartition('.')
            replacement = FineIntegerConv(conv, candidate_calibration[name]['quanta'], candidate_mode)
            replacement.capture_audit = capture_audit
            self.core.get_submodule(parent)._modules[leaf] = replacement
        self.candidate_mode = candidate_mode
        self.variant = 'combined_integer_' + candidate_mode + '_' + encoder_mode
        self.eval()

    def candidate_leaves(self):
        return {n: m for n, m in self.core.named_modules()
                if n.startswith(CANDIDATE_PREFIXES) and isinstance(m, FineIntegerConv)}

    def step_spatial(self, spatial, state, meta, detach_state=True, requests=None):
        if not detach_state or requests is not None:
            raise ValueError('Frozen whole stream only')
        reason, _ = reset_reason(state, meta)
        previous = {} if reason else state['candidate_integer']
        leaves = self.candidate_leaves()
        if set(previous) - set(leaves):
            raise ValueError('Unknown candidate execution state')
        for name, leaf in leaves.items():
            leaf.previous = previous.get(name)
            leaf.next_state = None
            leaf.statistics = {}
            leaf.audit = None
        try:
            pred, new, info = super().step_spatial(spatial, state, meta, detach_state, requests)
            executed = {n: m for n, m in leaves.items() if m.statistics}
            expected = {n for n in leaves if n.startswith('fine_update.') or not info['full_refresh']}
            if set(executed) != expected:
                raise RuntimeError('Candidate execution schedule mismatch')
            retained = dict(previous) if self.candidate_mode == 'delta' else {}
            for name, leaf in executed.items():
                if leaf.next_state is not None:
                    retained[name] = leaf.next_state
            new['candidate_integer'] = retained
            info['candidate_integer'] = {n: dict(m.statistics) for n, m in executed.items()}
            capacity = sum(v.numel() * v.element_size() for pair in retained.values() for v in pair)
            info['candidate_cache_bytes'] = capacity
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
        for leaf in self.candidate_leaves().values():
            leaf.audit = None
