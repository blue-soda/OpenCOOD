"""Factorial encoder/receiver transplant with a permanently frozen 3D encoder."""
from torch import nn
from opencood.models.e3dsnn_fusion import E3dsnnFusion
from opencood.models.e3dsnn_single_ann import replace_counts


class E3dsnnHybrid(E3dsnnFusion):
    def __init__(self, args):
        encoder = args['activation']
        receiver = args['receiver_activation']
        if encoder not in ('count4', 'relu') or receiver not in ('count4', 'relu'):
            raise ValueError('Unknown hybrid activation')
        if args.get('freeze_detector', False) or not args.get('freeze_detector_bn', False):
            raise ValueError('Hybrid protocol freezes BN and encoder, not the entire detector')
        if args.get('fusion_readout', 'activated') != 'activated':
            raise ValueError('Do not mix the continuous-readout ablation into transplants')
        # Start with Count4 everywhere, then replace only the requested modules.
        base = dict(args, activation='count4')
        super().__init__(base)
        self.activation = self.encoder_activation = encoder  # wire codec follows encoder
        self.receiver_activation = receiver
        if encoder == 'relu':
            replace_counts(self.backbone_3d)
        if receiver == 'relu':
            replace_counts(self.fusion_net)
            replace_counts(self.backbone_2d)
            self.fusion_activation = nn.ReLU(inplace=False)
        self.freeze_encoder = True
        for parameter in self.backbone_3d.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone_3d.eval()
        return self
