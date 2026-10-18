"""Fixed Count4 sender/ANN decoder; fusion activation x message support."""
from opencood.models.e3dsnn_budget import E3dsnnBudget
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
from opencood.models.fuse_modules.e3dsnn_message_support import message_support


class E3dsnnMessageFusion(E3dsnnBudget):
    def __init__(self, args):
        if args['activation'] != 'count4' or args['train_encoder']:
            raise ValueError('Message fusion protocol freezes the Count4 encoder')
        kind = args['fusion_activation_kind']
        support = args['message_support']
        if kind not in ('count4', 'relu') or support not in ('geometry', 'received'):
            raise ValueError('Unknown message fusion treatment')
        super().__init__(args)
        self.fusion_activation_kind = kind
        self.message_support_kind = support
        if kind == 'count4':
            self.fusion_net[2] = Multispike()
            self.fusion_activation = Multispike()

    def fusion_support(self, road, transform, coverage):
        return message_support(road, transform, coverage, self.origin, self.step,
                               self.message_support_kind)
