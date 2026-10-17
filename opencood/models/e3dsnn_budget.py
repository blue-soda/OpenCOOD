"""Sparse-message task adaptation with a frozen ANN BEV/continuous decoder."""
from opencood.models.e3dsnn_hybrid import E3dsnnHybrid
from opencood.utils.e3dsnn_budget import budget_features, decoder_digest
from opencood.utils.quantized_feature_packet import validate_options


class E3dsnnBudget(E3dsnnHybrid):
    def __init__(self, args):
        if args['receiver_activation'] != 'relu' or args.get('message_codec') is not None:
            raise ValueError('Budget adaptation requires ANN receiver and dedicated budget_codec')
        super().__init__(args)
        self.budget_codec = args['budget_codec']
        validate_options(self.budget_codec)
        self.train_encoder = bool(args['train_encoder'])
        self.freeze_encoder = not self.train_encoder
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith('fusion_net.') or name == 'fusion_scale' or
                                     (self.train_encoder and name.startswith('backbone_3d.')))

    def train(self, mode=True):
        super().train(mode)
        self.backbone_2d.eval()
        for head in (self.cls_head, self.reg_head, self.dir_head): head.eval()
        return self

    def process_road_features(self, sparse):
        # Wire path applies quantization once, directly to the original sender.
        if self.packet_roundtrip: return sparse
        features = budget_features(sparse.indices, sparse.features, self.activation, self.budget_codec)
        return sparse.replace_feature(features)

    def wire_options(self):
        return self.budget_codec

    def frozen_decoder_digest(self):
        return decoder_digest(self)
