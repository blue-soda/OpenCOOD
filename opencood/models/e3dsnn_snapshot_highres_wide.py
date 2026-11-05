"""High-resolution head with a wider ANN BEV backbone and unchanged head width.

Block filters grow while the deblock output channels stay (128, 128), so the
local-detail sum, the 256-channel head input and every downstream shape are
unchanged. The wider backbone starts with zero-padded kernels supplied by the
preparation script: extra input/output channels are exactly zero, so the model
is functionally identical to the base one before the first update.
"""
from opencood.models.e3dsnn_single import AttributeConfig
from opencood.models.e3dsnn_snapshot_highres import E3dsnnSnapshotHighres


class E3dsnnSnapshotHighresWide(E3dsnnSnapshotHighres):
    def __init__(self, args):
        super().__init__(args)
        opts = args.get('bev_wide', {})
        filters = tuple(int(x) for x in opts.get('filters', (96, 160)))
        upsample = tuple(int(x) for x in opts.get('upsample_filters', (128, 128)))
        if filters[0] <= 64 or filters[1] <= 128:
            raise ValueError('Wide BEV must exceed the base 64/128 filters')
        if upsample != (128, 128):
            raise ValueError('Head input must stay 256: upsample filters are fixed')
        in_ch = self.backbone_2d.blocks[0][2].in_channels
        cfg = AttributeConfig(LAYER_NUMS=[5, 5], LAYER_STRIDES=[1, 2],
                              NUM_FILTERS=list(filters),
                              UPSAMPLE_STRIDES=[1, 2],
                              NUM_UPSAMPLE_FILTERS=list(upsample))
        # Keep the exact backbone family of the base model (the snapshot path
        # replaces the spiking BEV with the small ANN BEV), so widening changes
        # capacity only, not the activation family.
        self.backbone_2d = type(self.backbone_2d)(cfg, in_ch)
        self.bev_filters = filters
