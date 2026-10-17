"""All-SNN receiver with fixed or learnable BEV quantization step sizes."""
from opencood.models.e3dsnn_hybrid import E3dsnnHybrid
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
from opencood.models.sub_modules.e3dsnn.bev_scale import ScaledCount4, scale_report


class E3dsnnBevScale(E3dsnnHybrid):
    def __init__(self, args):
        if args['activation'] != 'count4' or args['receiver_activation'] != 'count4':
            raise ValueError('BEV scale study requires Count4 encoder and receiver')
        self.bev_scale_mode = args['bev_scale_mode']
        if self.bev_scale_mode not in ('fixed', 'learned'):
            raise ValueError('Unknown BEV scale mode')
        super().__init__(args)
        self.fusion_activation_kind = 'count4'
        self.bev_activation = 'scaled_count4' if self.bev_scale_mode == 'learned' else 'count4'
        def replace(module):
            count = 0
            for name, child in list(module.named_children()):
                if isinstance(child, Multispike):
                    setattr(module, name, ScaledCount4(child, self.bev_scale_mode == 'learned'))
                    count += 1
                else: count += replace(child)
            return count
        if replace(self.backbone_2d) != 14:
            raise RuntimeError('Unexpected BEV neuron layout')

    def bev_scale_report(self):
        return scale_report(self.backbone_2d)
