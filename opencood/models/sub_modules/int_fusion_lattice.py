"""One fixed fusion input lattice and a per-output static/delta arithmetic audit."""
import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_integer_event_reference import integer_input


def neighborhoods(levels, measure):
    # One count per branch and output location, including boundary padding.
    if measure=='nonzero':
        cells=(levels!=0).sum(1)
    elif measure=='one':
        cells=(levels.abs()==1).sum(1)
    else:
        cells=levels.abs().sum(1)
    if int(cells.max())*9>=2**23:
        raise ValueError('Fusion tap sum outside exact pooling range')
    return (F.avg_pool2d(cells[:,None].float(),3,1,1)*9).round().long()


class FusionInputLattice:
    def __init__(self, quanta=None):
        self.quanta = None if quanta is None else torch.as_tensor(quanta,device='cuda',dtype=torch.float32).view(1,384,1,1)
        self.square_sum = None
        self.samples = 0
        self.previous = None

    def begin(self, reason):
        if reason is not None:
            self.previous=None

    def observe(self, module, inputs):
        x,=inputs
        if self.quanta is None:
            squared=x.double().square().sum(dim=(0,2,3))
            self.square_sum=squared if self.square_sum is None else self.square_sum+squared
            self.samples+=x.shape[0]*x.shape[2]*x.shape[3]
            return None
        n=integer_input(x,self.quanta)
        if int(n.abs().max())>=2**22:
            raise ValueError('Fusion code too large for exact integer difference')
        levels=n.int().reshape(3,128,*x.shape[-2:])
        static=neighborhoods(levels,'nonzero')
        static_one=neighborhoods(levels,'one')
        if self.previous is None:
            use_delta=torch.zeros_like(static,dtype=torch.bool)
            delta=delta_one=delta_units=torch.zeros_like(static)
        else:
            change=levels-self.previous
            delta=neighborhoods(change,'nonzero')
            delta_one=neighborhoods(change,'one')
            delta_units=neighborhoods(change,'units')
            use_delta=delta<static
        self.previous=levels
        def totals(v):
            return v.flatten(1).sum(1).tolist()
        # Counts are taps; each has 128 output-channel fanout.
        self.ledger=dict(static=totals(static),static_chosen=totals(static.masked_fill(use_delta,0)),
            delta_chosen=totals(delta.masked_fill(~use_delta,0)),delta_sites=totals(use_delta),
            one_static=totals(static_one.masked_fill(use_delta,0)),one_delta=totals(delta_one.masked_fill(~use_delta,0)),
            repeated_delta=totals(delta_units.masked_fill(~use_delta,0)),max_code=int(levels.abs().max()))
        return (n.float()*self.quanta,)

    def calibrated(self):
        return ((self.square_sum/self.samples).sqrt()*.125).clamp_min(1e-4).float().tolist()
