"""Inference-only sender-local confidence selection; extra ANN sender compute."""
import torch
from opencood.models.e3dsnn_budget import E3dsnnBudget
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike


class E3dsnnSenderRank(E3dsnnBudget):
    def __init__(self, args):
        if args['train_encoder'] or args.get('message_support', 'geometry') != 'geometry':
            raise ValueError('Fixed geometry-support inference probe only')
        super().__init__(args)
        self.sender_ranking = args['sender_ranking']
        self.fusion_activation_kind = args['fusion_activation_kind']
        if self.sender_ranking not in ('l1', 'confidence') or self.fusion_activation_kind not in ('relu', 'count4'):
            raise ValueError('Unknown sender ranking or fusion activation')
        if self.fusion_activation_kind == 'count4':
            self.fusion_net[2] = Multispike()
            self.fusion_activation = Multispike()
        for p in self.parameters(): p.requires_grad_(False)

    def packet_row_scores(self, sparse):
        if self.sender_ranking == 'l1': return None
        dense = sparse.dense()
        bev = self.backbone_2d({'spatial_features': dense.flatten(1, 2)})['spatial_features_2d']
        logits = self.cls_head(bev).amax(1)
        if logits.shape != (dense.shape[0], dense.shape[3], dense.shape[4]):
            raise ValueError('Confidence head must preserve sender XY grid')
        if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite sender logits')
        indices = sparse.indices.long()
        # Maximum anchor logit is monotonic in maximum sigmoid confidence,
        # without rounding all high-confidence logits into the same probability.
        return logits[indices[:, 0], indices[:, 2], indices[:, 3]].detach().cpu().numpy()

    def forward(self, data):
        if self.training or not self.packet_roundtrip:
            raise ValueError('Sender ranking probe is eval-only with actual wire messages')
        return super().forward(data)
