"""Fixed Count4 encoder, independently selected fusion and BEV activations."""
from torch import nn
from opencood.models.e3dsnn_hybrid import E3dsnnHybrid
from opencood.models.e3dsnn_single_ann import replace_counts


class E3dsnnReceiverSplit(E3dsnnHybrid):
    def __init__(self, args):
        if args['activation'] != 'count4':
            raise ValueError('Receiver split requires the shared frozen Count4 encoder')
        fusion = args['fusion_activation_kind']
        bev = args['bev_activation']
        if fusion not in ('count4', 'relu') or bev not in ('count4', 'relu'):
            raise ValueError('Unknown receiver component activation')
        super().__init__(dict(args, receiver_activation='count4'))
        self.fusion_activation_kind = fusion
        self.bev_activation = bev
        self.receiver_activation = fusion if fusion == bev else 'mixed'
        if fusion == 'relu':
            replace_counts(self.fusion_net)
            self.fusion_activation = nn.ReLU(inplace=False)
        if bev == 'relu':
            replace_counts(self.backbone_2d)
