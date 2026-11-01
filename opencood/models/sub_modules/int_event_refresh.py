"""Scalar refresh-policy primitive, not a spiking feature detector.

Input evidence must be a cheap causal signal calibrated on training data.
No detector integration, parameter calibration, sparse kernel or AP claim here.
"""
import math


class TemporalRefreshController:
    def __init__(self, mode='event', threshold=1., tau=.2, reference_dt=.1, max_gap=.25):
        if mode not in ('threshold','continuous','event'):
            raise ValueError('Unknown control mode')
        if any(not math.isfinite(v) or v<=0 for v in (threshold,tau,reference_dt,max_gap)):
            raise ValueError('Finite positive controller parameters required')
        self.mode=mode;self.threshold=threshold;self.tau=tau
        self.reference_dt=reference_dt;self.max_gap=max_gap

    def step(self, evidence, timestamp_us, scene, state=None, force_refresh=False):
        if type(timestamp_us) is not int or not math.isfinite(evidence) or evidence<0:
            raise ValueError('Integer timestamp and finite nonnegative evidence required')
        reason='initial' if state is None else None;dt=0.
        if state is not None:
            if state['scene']!=scene:reason='scene'
            else:
                dt=(timestamp_us-state['timestamp_us'])/1e6
                if dt<=0:raise ValueError('Noncausal controller input')
                if dt>self.max_gap:reason='gap'
            if not math.isfinite(state['membrane']) or state['membrane']<0:
                raise ValueError('Invalid membrane state')
        if force_refresh:reason='forced'
        if reason is not None:
            new=dict(membrane=0.,timestamp_us=timestamp_us,scene=scene)
            return True,new,dict(reason=reason,spike=False,pre_reset_membrane=0.,dt=dt)
        # Exact integration for a constant evidence rate over this interval.
        # This makes no-spike continuous integration invariant to subdividing dt.
        if self.mode=='threshold':potential=float(evidence)
        else:
            decay=math.exp(-dt/self.tau)
            drive=evidence*(self.tau/self.reference_dt)*(-math.expm1(-dt/self.tau))
            potential=state['membrane']*decay+drive
        if not math.isfinite(potential):raise ValueError('Nonfinite integrated evidence')
        triggered=potential>=self.threshold
        # Event arm subtracts one threshold, allowing residual charge to persist.
        # At most one refresh is issued per physical frame; there are no virtual steps.
        membrane=potential-self.threshold if self.mode=='event' and triggered else potential
        if self.mode=='threshold':membrane=0.
        new=dict(membrane=membrane,timestamp_us=timestamp_us,scene=scene)
        return triggered,new,dict(reason='evidence' if triggered else None,
            spike=self.mode=='event' and triggered,pre_reset_membrane=potential,dt=dt)


def replay_refresh_trace(rows, mode, threshold, tau=.2):
    """Replay own-anchor coverage from a causal pairwise geometry trace.

    Coarse states are independent of deep refresh decisions in this pyramid.
    Geometry is indexed by each candidate's actual anchor, never baseline flags.
    """
    policy=TemporalRefreshController(mode=mode,threshold=threshold,tau=tau)
    state=None;anchor=None;records=[]
    for index,row in enumerate(rows):
        forced=row['reset']
        if anchor is None:forced=forced or 'initial'
        elif forced is None:
            age=(row['timestamp_us']-rows[anchor]['timestamp_us'])/1e6
            if age>=.25-1e-6:forced='age'
            elif row['coverage_by_anchor'][str(anchor)]>0:forced='coverage'
        fired,state,info=policy.step(row['evidence'],row['timestamp_us'],row['segment'],state,forced is not None)
        reason=forced or ('evidence' if fired else None)
        if reason is not None:anchor=index
        records.append(dict(refresh=reason,anchor_index=anchor,membrane=state['membrane'],spike=info['spike']))
    return records
