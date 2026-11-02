"""Receive-only adaptation with a fixed SNN encoder and fixed event sender."""
from opencood.models.e3dsnn_snapshot_highres import E3dsnnSnapshotHighres


class E3dsnnSnapshotReceiverProbe(E3dsnnSnapshotHighres):
    def __init__(self, args):
        super().__init__(args)
        self.receiver_probe = args['receiver_probe']
        if self.receiver_probe not in ('membrane', 'prequantized', 'continuous'):
            raise ValueError('Unknown registered receiver probe')
        self.joint_receiver.snapshot_receive_mode = self.receiver_probe
        prefixes = ('content.expand.', 'joint_receiver.', 'backbone_2d.',
                    'local_detail.', 'cls_head.', 'reg_head.', 'dir_head.')
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
        self.freeze_encoder = True
        self.train_encoder = False
