"""Architecture-matched ANN control: replace Count4 with continuous ReLU.

Identical parameter names, shapes, initialization, sparse/BEV convolutions and
heads. This compares the complete bounded/discrete activation to ReLU; a later
clipped-ReLU control would isolate rounding from the upper-bound effect.
"""
from torch import nn
from opencood.models.e3dsnn_single import E3dsnnSingle
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike


def replace_counts(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Multispike):
            setattr(module, name, nn.ReLU(inplace=False))
            count += 1
        else:
            count += replace_counts(child)
    return count


class E3dsnnSingleAnn(E3dsnnSingle):
    def __init__(self, args):
        super().__init__(args)
        self.replaced_count_modules = replace_counts(self)
        if self.replaced_count_modules == 0:
            raise RuntimeError('ANN control did not replace any count activations')
