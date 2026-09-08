# -*- coding: utf-8 -*-
"""
Lightweight, opt-in diagnostics for training and validation.

The utilities here are deliberately defensive: diagnostics should help inspect
experiments, never break a long training run when visualization or scalar
serialization hits an unexpected dataset/model edge case.
"""

import json
import math
import os
import traceback

import torch

from opencood.visualization import simple_vis


class DiagnosticsManager(object):
    def __init__(self, hypes, save_dir):
        self.cfg = hypes.get('diagnostics', {}) or {}
        self.enabled = bool(self.cfg.get('enabled', False))
        self.save_dir = os.path.join(save_dir, 'diagnostics')
        self.scalar_interval = int(self.cfg.get('scalar_interval', 10))
        self.grad_interval = int(self.cfg.get('grad_interval', 0))
        self.val_visualize = self.cfg.get('val_visualize', {}) or {}
        self.val_vis_enabled = bool(self.val_visualize.get('enabled', False))
        self.val_vis_interval = int(self.val_visualize.get('interval', 0))
        self.max_val_vis = int(self.val_visualize.get('max_images', 0))
        self.val_vis_method = self.val_visualize.get('method', 'bev')
        self.val_vis_count = 0

        if self.enabled:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, 'diagnostics_config.json'), 'w') as f:
                json.dump(self.cfg, f, indent=2, sort_keys=True)

    @staticmethod
    def _is_scalar(value):
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if torch.is_tensor(value):
            return value.numel() == 1
        return False

    @staticmethod
    def _to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().float().cpu().item())
        return float(value)

    def _should_log(self, step, interval):
        return self.enabled and interval > 0 and step % interval == 0

    def log_train_step(self, writer, epoch, batch_id, batch_len, loss, output_dict, optimizer):
        if not self._should_log(epoch * batch_len + batch_id, self.scalar_interval):
            return
        step = epoch * batch_len + batch_id
        writer.add_scalar('Diagnostics/train_total_loss', self._to_float(loss), step)
        for group_id, group in enumerate(optimizer.param_groups):
            writer.add_scalar('Diagnostics/lr_group_%d' % group_id, group['lr'], step)
        self.log_output_scalars(writer, output_dict, step, prefix='Diagnostics/output')

    def log_validation_step(self, writer, epoch, batch_id, batch_len, loss, output_dict):
        if not self._should_log(epoch * batch_len + batch_id, self.scalar_interval):
            return
        step = epoch * batch_len + batch_id
        writer.add_scalar('Diagnostics/val_total_loss', self._to_float(loss), step)
        self.log_output_scalars(writer, output_dict, step, prefix='Diagnostics/val_output')

    def log_output_scalars(self, writer, output_dict, step, prefix):
        for key, value in sorted(output_dict.items()):
            if self._is_scalar(value):
                writer.add_scalar('%s/%s' % (prefix, key), self._to_float(value), step)

    def log_gradients(self, writer, model, epoch, batch_id, batch_len):
        step = epoch * batch_len + batch_id
        if not self._should_log(step, self.grad_interval):
            return
        total_sq_norm = 0.0
        trainable_count = 0
        for _, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            grad_norm = parameter.grad.detach().data.norm(2).item()
            total_sq_norm += grad_norm * grad_norm
            trainable_count += 1
        writer.add_scalar('Diagnostics/grad_total_norm', total_sq_norm ** 0.5, step)
        writer.add_scalar('Diagnostics/grad_parameter_count', trainable_count, step)

    def maybe_save_validation_visual(self, batch_data, output_dict, dataset, hypes, epoch, batch_id):
        if not self.enabled or not self.val_vis_enabled:
            return
        if self.val_vis_interval <= 0 or self.max_val_vis <= 0:
            return
        if self.val_vis_count >= self.max_val_vis:
            return
        if batch_id % self.val_vis_interval != 0:
            return

        try:
            ego_content = dict(batch_data['ego'])
            matrix_source = ego_content['anchor_box']
            if 'transformation_matrix' not in ego_content:
                ego_content['transformation_matrix'] = torch.eye(
                    4, device=matrix_source.device, dtype=torch.float32)
            if 'transformation_matrix_clean' not in ego_content:
                ego_content['transformation_matrix_clean'] = torch.eye(
                    4, device=matrix_source.device, dtype=torch.float32)
            post_data = {'ego': ego_content}
            pred_box_tensor, _, gt_box_tensor = dataset.post_process(
                post_data, {'ego': output_dict})
            if pred_box_tensor is None:
                return
            vis_dir = os.path.join(self.save_dir, 'val_visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(
                vis_dir, 'epoch_%03d_batch_%05d_%s.png' %
                (epoch, batch_id, self.val_vis_method))
            simple_vis.visualize(
                pred_box_tensor,
                gt_box_tensor,
                batch_data['ego']['origin_lidar'][0],
                hypes['postprocess']['gt_range'],
                vis_path,
                method=self.val_vis_method)
            self.val_vis_count += 1
        except Exception as exc:
            warning_path = os.path.join(self.save_dir, 'visualization_warnings.log')
            with open(warning_path, 'a') as f:
                f.write('epoch=%s batch=%s error=%s\n' % (epoch, batch_id, repr(exc)))
                f.write(traceback.format_exc())
                f.write('\n')
