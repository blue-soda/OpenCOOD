"""Inference-only regional updates of the migrated local high-level state."""
import math
import torch
import torch.nn.functional as F
from opencood.models.sub_modules.int_local_state import LocalStateDetector
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cache_repair import refresh_tiles, expand_tiles


class ComputedUpdate(torch.nn.Module):
    """Only for an explicitly expensive selector diagnostic; no second U call."""
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.eval()

    def forward(self, unused):
        return self.value


def relative_change_score(current, reference, tile):
    """Per-tile relative energy change, also used as a training-only target."""
    pool = lambda x: F.avg_pool2d(x, tile, tile, ceil_mode=True, count_include_pad=False)
    error = pool((current-reference).square().mean(1, keepdim=True))
    energy = torch.maximum(pool(current.square().mean(1, keepdim=True)),
                           pool(reference.square().mean(1, keepdim=True)))
    return error / energy.clamp_min(1e-6)


def score_request(score, valid, age, fraction, tile, max_age):
    chosen = torch.zeros_like(score, dtype=torch.bool)
    count = int(math.ceil(fraction * score.numel()))
    if fraction == 1:
        chosen.fill_(True)
    elif count:
        indices = score.flatten().topk(count, sorted=False).indices
        chosen.view(-1)[indices] = score.flatten()[indices] > 0
    # Mandatory coverage/age is additional to the nominal ranking budget.
    return expand_tiles(chosen, tile, valid.shape[-2:]) | ~valid | (age >= max_age-1e-6)


def change_request(current, reference, valid, age, fraction, tile, max_age):
    """Rank tiles by relative shallow-feature change since their last refresh."""
    return score_request(relative_change_score(current, reference, tile),
                         valid, age, fraction, tile, max_age)


class RegionalLocalStateDetector(LocalStateDetector):
    """Same weights as B; unselected cells retain their warped high-level state.

    The extra shallow reference and age follow the same nearest-neighbor warp as
    H. Reference cells change only when their corresponding H cells are written,
    so small successive input changes accumulate until a refresh.
    """
    def __init__(self, teacher, config, fraction=.25, tile=16, max_age=.5,
                 execution='sparse', fallback_fraction=.85, batch_size=8, selector='shallow',
                 selector_net=None):
        if not 0 <= fraction <= 1 or tile < 1 or not 0 < max_age < float('inf'):
            raise ValueError('Invalid regional update policy')
        if execution not in ('sparse', 'masked_dense', 'fixed_sparse') or not 0 < fallback_fraction <= 1 or batch_size < 1:
            raise ValueError('Invalid execution policy')
        if selector not in ('shallow', 'oracle_state', 'learned'):
            raise ValueError('Unknown selector')
        if (selector == 'learned') != (selector_net is not None):
            raise ValueError('Learned selector requires its explicit network')
        self.fraction, self.tile, self.max_age = fraction, tile, max_age
        self.execution, self.fallback_fraction = execution, fallback_fraction
        self.batch_size = batch_size
        self.selector = selector
        super().__init__(teacher, config, 'recurrent')
        if selector_net is not None:
            self.selector_net = selector_net
        self.collect_selector_targets = False
        self.selector_example = None

    def select_request(self, packed, old, evidence, reference, coverage, age):
        candidate = self.update(packed) if (
            self.selector == 'oracle_state' or self.collect_selector_targets) else None
        if self.selector == 'learned':
            # Training collects full-U labels separately. Neither candidate nor
            # labels enter the scorer's inputs or the deployment path.
            with torch.set_grad_enabled(self.collect_selector_targets):
                logits = self.selector_net(evidence.detach(), reference.detach(),
                                           old.detach(), coverage, age.detach(), self.tile)
            if self.collect_selector_targets:
                target = relative_change_score(candidate.detach(), old.detach(), self.tile)
                self.selector_example = (logits, (target+1e-4).log().detach())
            score = logits.detach().clamp(-16, 8).exp()
        else:
            current, previous = (candidate, old) if candidate is not None else (evidence, reference)
            score = relative_change_score(current, previous, self.tile)
        return score_request(score, coverage, age, self.fraction, self.tile, self.max_age), candidate

    def compute_update(self, packed, old, valid, state, meta, lattice, reason):
        if self.training:
            raise ValueError('Regional cache is inference-only; no cache-aware training yet')
        if self.collect_selector_targets and self.selector != 'learned':
            raise ValueError('Target collection is only for learned selector training')
        self.selector_example = None
        evidence = packed[:, :128]
        if reason is not None:
            value = self.update(packed)
            written = torch.ones_like(valid)
            reference = evidence
            age = torch.zeros_like(evidence[:, :1])
            stats = dict(requested_fraction=1., selected_fraction=1.,
                         executed_input_area_ratio=1., actual_write_fraction=1.,
                         crop_area_ratio_before_fallback=1., tiles=1, fallback=False)
            mandatory = 1.
        else:
            auxiliary, aux_valid = warp_bev(
                torch.cat([state['evidence_cache'], state['cache_age']], 1),
                state['pose'], meta['pose'], lattice)
            reference = auxiliary[:, :128]
            dt = (int(meta['timestamp_us'])-state['timestamp_us'])/1e6
            age = auxiliary[:, 128:] + dt
            coverage = valid & aux_valid
            requested, candidate = self.select_request(packed, old, evidence, reference, coverage, age)
            mandatory = float((~coverage | (age >= self.max_age-1e-6)).float().mean())
            value, written, stats = refresh_tiles(
                self.update if candidate is None else ComputedUpdate(candidate),
                packed, old, requested, 1, self.update.radius,
                tile=self.tile, fallback_fraction=self.fallback_fraction,
                execution=self.execution if candidate is None else 'masked_dense', batch_size=self.batch_size)
        reference = torch.where(written, evidence, reference).detach()
        age = torch.where(written, torch.zeros_like(age), age).detach()
        stats.update(mandatory_fraction=mandatory, max_age_after_update=float(age.max()))
        stats['oracle_full_update'] = float(self.selector == 'oracle_state')
        stats['training_full_update'] = float(self.collect_selector_targets)
        if self.selector == 'oracle_state' or self.collect_selector_targets:
            stats['executed_input_area_ratio'] = 1.
        size = sum(x.numel()*x.element_size() for x in (value, reference, age))
        return value, dict(evidence_cache=reference, cache_age=age), dict(
            dense_update=False, cache_bytes=size, state_bytes=size, region_cache=stats)
