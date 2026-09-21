"""Opt-in CoBEVFlow model using the separate GPU ROI post-processing path.

The network, weights, fusion module and matcher are inherited unchanged from
the baseline.  Only the per-frame ROI decode/NMS helper is replaced.  This
module is selected by a new config's ``model.core_method`` and never changes
the baseline model file.
"""

import os

import torch

from opencood.models.point_pillar_codyntrust_cobevflow_baseline import (
    PointPillarCodyntrustCobevflowBaseline,
    make_roi_cache_key,
    tensor_digest,
    value_digest,
)
from opencood.tools.cobevflow_optimized_roi import (
    cobevflow_fast_single_post_process,
)


def _max_abs_result_delta(reference, candidate):
    if reference.keys() != candidate.keys():
        return False, 'top-level keys differ'
    max_delta = 0.0
    for key in reference:
        if isinstance(reference[key], dict):
            ok, detail = _max_abs_result_delta(reference[key], candidate[key])
            if not ok:
                return False, '{}: {}'.format(key, detail)
            continue
        if torch.is_tensor(reference[key]):
            if reference[key].shape != candidate[key].shape:
                return False, '{} shape {} != {}'.format(
                    key, tuple(reference[key].shape), tuple(candidate[key].shape))
            if reference[key].numel():
                delta = (reference[key].detach().float() -
                         candidate[key].detach().float()).abs().max().item()
                max_delta = max(max_delta, delta)
        elif reference[key] != candidate[key]:
            return False, '{} values differ'.format(key)
    return True, 'max_abs_delta={:.6g}'.format(max_delta)


class PointPillarCodyntrustCobevflowOptimized(
        PointPillarCodyntrustCobevflowBaseline):
    """Drop-in optimized CoBEVFlow model with opt-in equivalence checking."""

    def _generate_pred_bbx_frames_for_roi(self, dataset, m_single,
                                          trans_mat_pastk_2_past0,
                                          past_time_diff, anchor_box,
                                          sample_idx=None, cav_idx=0):
        post_processor = getattr(dataset, 'post_processor', None)
        params = getattr(post_processor, 'params', {}) \
            if post_processor is not None else {}
        target_args = params.get('target_args', None)
        old_threshold = None
        if target_args is not None:
            old_threshold = target_args.get('score_threshold', None)
            if self.roi_score_threshold is not None:
                target_args['score_threshold'] = float(self.roi_score_threshold)

        split = 'train' if getattr(dataset, 'train', False) else 'val'
        threshold = self.roi_score_threshold
        if threshold is None and target_args is not None:
            threshold = target_args.get('score_threshold', None)
        prediction_hash = self._roi_prediction_hash(m_single)
        cache_key = make_roi_cache_key(
            sample_idx=sample_idx,
            cav_idx=cav_idx,
            split=split,
            k=self.k,
            threshold=threshold,
            num_roi_thres=self.num_roi_thres,
            time_diff=past_time_diff,
            prediction_hash=prediction_hash)
        cached = self.roi_cache.load(split, cache_key, anchor_box.device)
        if cached is not None:
            return cached

        try:
            result = cobevflow_fast_single_post_process(
                post_processor, m_single, trans_mat_pastk_2_past0,
                past_time_diff, anchor_box, self.k, self.num_roi_thres)

            # Equivalence checking is disabled for timing.  Set
            # COBEVFLOW_FAST_VERIFY=1 for a calibration run; by default a
            # mismatch falls back to the reference implementation so the
            # optimized model remains semantically safe during validation.
            if os.environ.get('COBEVFLOW_FAST_VERIFY', '').lower() in (
                    '1', 'true', 'yes'):
                reference = dataset.generate_pred_bbx_frames(
                    m_single, trans_mat_pastk_2_past0, past_time_diff,
                    anchor_box)
                equivalent, detail = _max_abs_result_delta(reference, result)
                if not equivalent:
                    print('[CoBEVFlowFastVerify] mismatch: {}'.format(detail))
                    if os.environ.get(
                            'COBEVFLOW_FAST_FALLBACK', '1').lower() in (
                                '1', 'true', 'yes'):
                        result = reference
                else:
                    print('[CoBEVFlowFastVerify] equivalent: {}'.format(detail))

            self.roi_cache.save(split, cache_key, result, {
                'sample_idx': sample_idx,
                'cav_idx': int(cav_idx),
                'threshold': threshold,
                'k': self.k,
                'num_roi_thres': self.num_roi_thres,
                'prediction_hash': prediction_hash,
                'implementation': 'cobevflow_optimized_gpu_roi_v1',
            })
            return result
        finally:
            if old_threshold is not None:
                target_args['score_threshold'] = old_threshold
