"""Fine-BEV variant with wider physical context in the later BEV stage.

The fine grid halves the physical extent of every 3x3 kernel, so the later BEV
stage sees less context in metres even though the detection grid is finer.
This variant keeps the output grid, channels, branches and message path
unchanged and only swaps every other 3x3 stride-1 convolution in the second BEV
stage for a dilation-2 copy (same weights, same output size). It tests context
coverage rather than capacity: no parameters are added.
"""
from torch import nn

from opencood.models.e3dsnn_snapshot_fine_bev import E3dsnnSnapshotFineBev


class E3dsnnSnapshotFineBevDilated(E3dsnnSnapshotFineBev):
    def __init__(self, args):
        super().__init__(args)
        stage = self.backbone_2d.blocks[1]
        convs = [m for m in stage.modules()
                 if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3) and m.stride == (1, 1)]
        if not convs:
            raise ValueError('No stride-1 3x3 convolution found in the second BEV stage')
        self.dilated_convs = 0
        for conv in convs[::2]:
            conv.dilation = (2, 2)
            conv.padding = (2, 2)
            self.dilated_convs += 1
        if self.dilated_convs == 0:
            raise ValueError('Dilated variant changed no convolution')
