"""Bounded source-only linear probe; not a claimed final SNN architecture."""
import numpy as np
from opencood.utils import task_snapshot_packet as wire


def fit_probe(features, targets, observed):
    x=np.asarray(features,dtype=np.float64);y=np.asarray(targets);mask=np.asarray(observed,dtype=bool)
    if x.ndim!=2 or y.shape!=mask.shape or y.shape!=(len(x),32) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid teacher feature/target shapes')
    mean=x.mean(0);std=x.std(0);std[std<1e-6]=1.
    z=np.column_stack(((x-mean)/std,np.ones(len(x))))
    weights=np.zeros((z.shape[1],32));samples=[]
    penalty=np.eye(z.shape[1])*1e-3;penalty[-1,-1]=0
    for channel in range(32):
        use=mask[:,channel] & (y[:,channel]!=0);a=z[use];b=np.sign(y[use,channel]);samples.append(len(b))
        if not len(b):continue
        # Equal total weight for each sign when both signs are represented.
        positive=(b>0);w=np.ones(len(b))/len(b)
        if positive.any() and (~positive).any():
            w[positive]=.5/positive.sum();w[~positive]=.5/(~positive).sum()
        weights[:,channel]=np.linalg.solve(a.T@(a*w[:,None])+penalty,a.T@(b*w))
    return dict(mean=mean,std=std,weights=weights,samples=np.asarray(samples))


def source_scores(features,probe):
    x=np.asarray(features,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(probe['mean']) or not np.isfinite(x).all():raise ValueError('Invalid source features')
    z=np.column_stack(((x-probe['mean'])/probe['std'],np.ones(len(x))))
    return z@probe['weights']


def select_suppressed(full,addresses,accepted,removed,meta,budget=2048):
    """Suppress last copies inside original prefix, then refill under real bytes.

    No selected count is subtracted at the receiver. The sender edits its event
    stream and runs the ordinary codec admission again, including all metadata.
    """
    ids=np.asarray(addresses);drop=np.asarray(removed)
    if drop.shape!=np.asarray(full).shape or np.any(drop<0) or np.any(drop!=np.floor(drop)) or not 0<=accepted<=len(ids):
        raise ValueError('Invalid event suppression')
    prefix=np.bincount(ids[:accepted].astype(np.int64),minlength=drop.size).reshape(drop.shape)
    if np.any(drop>prefix):raise ValueError('Suppression exceeds paid original prefix')
    remaining=drop.ravel().astype(np.int64).copy();keep=np.ones(len(ids),dtype=bool)
    for i in range(accepted-1,-1,-1):
        address=ids[i]
        if remaining[address]:keep[i]=False;remaining[address]-=1
    assert not remaining.any()
    chosen=ids[keep];modified=np.bincount(chosen.astype(np.int64),minlength=drop.size).reshape(drop.shape)
    assert np.array_equal(modified,np.asarray(full)-drop)
    packet,counts,record=wire.select(modified,np.ones(32,dtype=np.float32),meta,budget,addresses=chosen)
    assert len(packet)<=budget
    return packet,counts,record
