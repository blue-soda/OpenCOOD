"""Scheme-2 joint task adaptation; unchanged IF8 snapshot inference graph.

Only the training scope/budget distribution and local auxiliary loss differ
from the preserved snapshot baseline. This is not a new spiking mechanism.
"""
import hashlib
import random
import torch
from opencood.models.e3dsnn_snapshot_if import E3dsnnSnapshotIf


JOINT_PREFIXES = ('backbone_3d.conv4.', 'backbone_3d.conv_out.',
                  'backbone_2d.', 'cls_head.', 'reg_head.', 'dir_head.',
                  'content.project.', 'content.expand.', 'joint_receiver.')


class E3dsnnSnapshotJoint(E3dsnnSnapshotIf):
    def __init__(self, args):
        super().__init__(args)
        opts = args['snapshot_joint']
        self.joint_train_budgets = tuple(opts['train_budgets'])
        if not self.joint_train_budgets or any(type(b) is not int or b < 2048
                                             for b in self.joint_train_budgets):
            raise ValueError('Expected positive actual byte budgets >= 2048')
        self.ego_aux_weight = float(opts['ego_aux_weight'])
        if self.ego_aux_weight != .5:
            raise ValueError('Registered matched auxiliary weight is 0.5')
        self.enable_ego_aux = True
        self.freeze_encoder = self.freeze_detector = False
        self.freeze_detector_bn = True
        self.train_encoder = True
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(JOINT_PREFIXES))
        self._local_bev = None

    def train(self, mode=True):
        # Explicitly override historical frozen-detector train() policies.
        torch.nn.Module.train(self, mode)
        for module in self.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        # Refresh checkpoint identity at each validation/deployment boundary.
        # Training packets are sizing surrogates, not exported wire evidence.
        self._encoder_identity = None
        return self

    def frozen_decoder_digest(self):
        # The backend is now trainable; the generic runner checks the complete
        # frozen-state digest below instead of its old frozen-backend contract.
        return None

    def fixed_state_digest(self):
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            if name not in trainable:
                digest.update(name.encode('utf8'))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def encode_membrane(self, lidar, batch_size):
        sparse, membrane = super().encode_membrane(lidar, batch_size)
        if self.training:
            self._local_bev = sparse.dense().flatten(1, 2)
        return sparse, membrane

    def forward(self, data):
        original_budget = self.snapshot_budget
        if self.training:
            # Consume one Python RNG draw in both arms. Worker augmentations
            # and torch dropout RNG remain matched; runner saves RNG on resume.
            self.snapshot_budget = self.joint_train_budgets[
                min(int(random.random() * len(self.joint_train_budgets)),
                    len(self.joint_train_budgets) - 1)]
        try:
            output = super().forward(data)
            if self.training:
                output['ego_output'] = self.detect_bev(self._local_bev)
            return output
        finally:
            self.snapshot_budget = original_budget
            self._local_bev = None
