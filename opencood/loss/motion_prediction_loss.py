import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionPredictionLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_type = args.get('loss_type', 'smooth_l1')
        self.beta = args.get('beta', 1.0)
        self.weight = args.get('weight', 1.0)
        self.loss_dict = {}

    def forward(self, output_dict, target_dict=None, mode=None, suffix=""):
        pred = output_dict['preds_coop']
        target = output_dict['gt_coop'].to(device=pred.device, dtype=pred.dtype)
        valid = torch.isfinite(pred).all(dim=-1) & torch.isfinite(target).all(dim=-1)
        if valid.sum() == 0:
            loss = pred.sum() * 0
        elif self.loss_type == 'mse':
            loss = F.mse_loss(pred[valid], target[valid])
        else:
            loss = F.smooth_l1_loss(
                pred[valid], target[valid], beta=self.beta)
        loss = loss * self.weight
        abs_err = torch.abs(pred[valid] - target[valid]).mean() \
            if valid.sum() > 0 else pred.sum() * 0
        self.loss_dict = {
            'total_loss': loss.detach(),
            'motion_loss': loss.detach(),
            'motion_abs_err': abs_err.detach(),
            'motion_valid_count': int(valid.sum().item()),
        }
        return loss

    def logging(self, epoch, batch_id, batch_len, writer, suffix=""):
        step = epoch * batch_len + batch_id
        for key, value in self.loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            writer.add_scalar(key + suffix, value, step)
