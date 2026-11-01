"""Cheap coarse-state evidence drives a scalar controller, with own anchors."""
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_event_refresh import TemporalRefreshController


def coarse_change(current, old, valid, reset):
    if reset:return 0.
    mask=valid.to(current.dtype)
    return float(((current-old).abs()*mask).sum()/((current.abs()*mask).sum()+1e-6))


class EvidenceTraceAnchor(AnchoredLatePyramidDetector):
    def observe_coarse(self, current, old, valid, reset):
        return coarse_change(current,old,valid,reset)


class EventAnchoredPyramidDetector(EvidenceTraceAnchor):
    def __init__(self, base, mode, threshold, tau=.2):
        super().__init__(base,period=2,max_age=.25,coverage='strict')
        self.controller_policy=TemporalRefreshController(mode=mode,threshold=threshold,tau=tau)

    def decide_refresh(self, state, meta, reason, age, uncovered, count, evidence):
        forced=reason
        if forced is None and age>=self.max_age-1e-6:forced='age'
        if forced is None and uncovered>0:forced='coverage'
        old=None if state is None else state['controller']
        triggered,new,info=self.controller_policy.step(evidence,int(meta['timestamp_us']),meta['segment'],old,
                                                       force_refresh=forced is not None)
        refresh=forced or ('evidence' if triggered else None)
        info.update(mode=self.controller_policy.mode,forced=forced,scalar_state_count=1)
        return refresh,new,info
