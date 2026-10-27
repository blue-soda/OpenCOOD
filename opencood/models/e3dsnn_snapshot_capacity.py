"""Matched small/wide ANN BEV comparison with frozen SNN/wire/receiver."""
from opencood.models.e3dsnn_snapshot_joint import E3dsnnSnapshotJoint
from opencood.utils.e3dsnn_capacity import widen_backend


class E3dsnnSnapshotCapacity(E3dsnnSnapshotJoint):
    def __init__(self,args):
        super().__init__(args)
        self.capacity_width=int(args['capacity']['bev_width'])
        if self.capacity_width not in (1,2):raise ValueError('Registered first comparison uses width 1 or 2')
        if args['receiver_activation']!='relu':raise ValueError('Capacity comparison retains ANN BEV')
        if self.capacity_width==2:widen_backend(self)
        self.freeze_encoder=True;self.train_encoder=False
        for name,p in self.named_parameters():
            p.requires_grad_(name.startswith(('backbone_2d.','cls_head.','reg_head.','dir_head.')))
