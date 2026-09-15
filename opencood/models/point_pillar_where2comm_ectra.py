# -*- coding: utf-8 -*-
# Author: Sizhe Wei <sizhewei@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.codyntrust_base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.decoder_domain_norm import split_decoder_statistics, set_decoder_domain
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.fuse_modules.raindrop_attn import raindrop_fuse as raindrop_attn_fuse
from opencood.models.fuse_modules.raindrop_flow import raindrop_fuse as raindrop_flow_fuse
from opencood.models.fuse_modules.raindrop_swin import raindrop_swin
from opencood.models.fuse_modules.raindrop_swin_w_single import raindrop_swin_w_single
from opencood.models.sub_modules.ectra_recurrent_alignment import EctraRecurrentAlignment
from opencood.models.sub_modules.ectra_roi_flow_refiner import EctraRoiFlowRefiner
from opencood.tools.matcher import Matcher
from collections import OrderedDict
import torch
from scipy.optimize import linear_sum_assignment

import numpy as np
from opencood.utils import box_utils
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world

def generate_noise(pos_std, rot_std, pos_mean=0, rot_mean=0):
    """ Add localization error to the 6dof pose
        Noise includes position (x,y) and rotation (yaw).
        We use gaussian distribution to generate noise.

    Args:

        pos_std : float 
            std of gaussian dist, in meter

        rot_std : float
            std of gaussian dist, in degree

        pos_mean : float
            mean of gaussian dist, in meter

        rot_mean : float
            mean of gaussian dist, in degree

    Returns:
        pose_noise: np.ndarray, [6,]
            [x, y, z, roll, yaw, pitch]
    """

    xy = np.random.normal(pos_mean, pos_std, size=(2))
    yaw = np.random.normal(rot_mean, rot_std, size=(1))

    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])

    return pose_noise

def get_past_k_pairwise_transformation2ego(past_k_lidar_pose, noise_level, k=3, max_cav=5):
    """
    Get transformation matrixes accross different agents to curr ego at all past timestamps.

    Parameters
    ----------
    base_data_dict : dict
        Key : cav id, item: transformation matrix to ego, lidar points.
    
    ego_pose : list
        ego pose

    max_cav : int
        The maximum number of cav, default 5

    Return
    ------
    pairwise_t_matrix : np.array
        The transformation matrix each cav to curr ego at past k frames.
        shape: (L, k, 4, 4), L is the max cav number in a scene, k is the num of past frames
        pairwise_t_matrix[i, j] is T i_to_ego at past_j frame
    """
    pos_std = noise_level['pos_std']
    rot_std = noise_level['rot_std']
    pos_mean = 0 
    rot_mean = 0
    
    pairwise_t_matrix = np.tile(np.eye(4), (max_cav, k, 1, 1)) # (L, k, 4, 4)

    ego_pose = past_k_lidar_pose[0, 0]

    t_list = []

    # save all transformation matrix in a list in order first.
    for cav_id in range(past_k_lidar_pose.shape[0]):
        past_k_poses = []
        for time_id in range(k):
            loc_noise = generate_noise(pos_std, rot_std)
            past_k_poses.append(x_to_world(past_k_lidar_pose[cav_id, time_id].cpu().numpy()+loc_noise))
        t_list.append(past_k_poses) # Twx
    
    ego_pose = x_to_world(ego_pose.cpu().numpy())
    for i in range(len(t_list)): # different cav
        if i!=0 :
            for j in range(len(t_list[i])): # different time
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                t_matrix = np.linalg.solve(t_list[i][j], ego_pose)  # Tjw*Twi = Tji
                pairwise_t_matrix[i, j] = t_matrix
    pairwise_t_matrix = torch.tensor(pairwise_t_matrix).to(past_k_lidar_pose.device)
    return pairwise_t_matrix


class PointPillarWhere2commEctra(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commEctra, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone'] and args['base_bev_backbone']['resnet']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        if 'num_sweep_frames' in args:    # number of frames we use in LSTM
            self.k = args['num_sweep_frames']
        else:
            self.k = 0

        if 'time_delay' in args:          # number of time delay
            self.tau = args['time_delay'] 
        else:
            self.tau = 0
        
        self.noise_flag = 0
        if 'noise' in args.keys():
            self.noise_flag = True
            self.noise_level = {'pos_std': args['noise']['pos_std'], 'rot_std': args['noise']['rot_std'], 'pos_mean': args['noise']['pos_mean'], 'rot_mean': args['noise']['rot_mean']}
            print(f'=== noise level : {self.noise_level} ===')

        self.dcn = False
        # if 'dcn' in args:
        #     self.dcn = True
        #     self.dcn_net = DCNNet(args['dcn'])

        self.design_mode = args.get('design_mode', 0)
        self.num_roi_thres = args.get('num_roi_thres', -1)
        self.viz_bbx_flag = args.get('viz_bbx_flag', False)
        self.ectra_roi_flag = args.get('ectra_roi', {}).get('enabled', False)
        self.ectra_dense_enabled = args.get('ectra', {}).get(
            'enabled', not self.ectra_roi_flag)
        self.ectra_roi_context_dim = int(args.get('ectra', {}).get(
            'roi_context_dim', 0))
        self.ectra_roi_match_thresh = float(args.get('ectra', {}).get(
            'roi_match_thresh', 5.0))
        self.ectra_roi_velocity_scale = float(args.get('ectra', {}).get(
            'roi_velocity_scale', 10.0))
        self.ectra_roi_flow_scale = float(args.get('ectra', {}).get(
            'roi_flow_scale', 20.0))
        self.ectra_roi_score_threshold = args.get('ectra', {}).get(
            'roi_score_threshold', None)
        self.ectra_roi_fallback_score_threshold = args.get('ectra', {}).get(
            'roi_fallback_score_threshold', None)
        self.ectra_roi_min_count = int(args.get('ectra', {}).get(
            'roi_min_count', 0))
        self.ectra_roi_context_topk = int(args.get('ectra', {}).get(
            'roi_context_topk', -1))
        diagnostics_args = args.get('diagnostics', {})
        self.diagnostic_roi_box_stats = bool(
            diagnostics_args.get('roi_box_stats', False))
        self.diagnostic_roi_match_distance = float(
            diagnostics_args.get('roi_match_distance', 4.0))
        x_span = max(float(args['lidar_range'][3] - args['lidar_range'][0]), 1.0)
        y_span = max(float(args['lidar_range'][4] - args['lidar_range'][1]), 1.0)
        self.ectra_roi_position_scale = (x_span * 0.5, y_span * 0.5)
        self.ectra_roi_size_scale = float(args.get('ectra', {}).get(
            'roi_size_scale', 8.0))

        # 用于 single 部分的监督
        self.single_supervise = False
        if 'with_compensation' in args and args['with_compensation']:
            self.compensation = True
            if 'with_single_supervise' in args and args['with_single_supervise']:
                self.rain_fusion = raindrop_swin_w_single(args['rain_model'])
                self.single_supervise = True
            else:
                self.rain_fusion = raindrop_swin(args['rain_model'])
        else: 
            self.compensation = False
            if self.ectra_roi_flag:
                self.rain_fusion = raindrop_flow_fuse(
                    args['rain_model'], self.design_mode)
            else:
                self.rain_fusion = raindrop_attn_fuse(args['rain_model'])

        self.multi_scale = args['rain_model']['multi_scale']
        self.ectra = EctraRecurrentAlignment(args.get('ectra', {}))
        self.ectra_roi = EctraRoiFlowRefiner(args.get('ectra_roi', {})) \
            if self.ectra_roi_flag else None
        self.matcher = Matcher('flow') if self.ectra_roi_flag else None
        
        if self.shrink_flag:
            dim = args['shrink_header']['dim'][0]
            self.cls_head = nn.Conv2d(int(dim), args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(int(dim), 7 * args['anchor_number'],
                                    kernel_size=1)
            self.fused_cls_head = nn.Conv2d(int(dim), args['anchor_number'],
                                    kernel_size=1)
            self.fused_reg_head = nn.Conv2d(int(dim), 7 * args['anchor_number'],
                                    kernel_size=1)
        else:
            self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                    kernel_size=1)

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2， # 384
            self.fused_dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'],
                                    kernel_size=1)
        else:
            self.use_dir = False

        self.independent_fusion_backbone = args.get('independent_fusion_backbone', False)
        if self.independent_fusion_backbone:
            if not self.ectra_roi_flag or not self.multi_scale:
                raise ValueError('Independent ECTRA fusion backbone requires multiscale ROI mode')
            self.fused_backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
            if self.shrink_flag:
                self.fused_shrink_conv = DownsampleConv(args['shrink_header'])
        if args.get('separate_decoder_bn', False):
            split_decoder_statistics(self.backbone.deblocks)
            if self.shrink_flag:
                split_decoder_statistics(self.shrink_conv)
        self.freeze_single_bn = args.get('freeze_single_bn', False)
        if args['backbone_fix']:
            self.backbone_fix()

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_single_bn:
            for name in ('pillar_vfe', 'backbone', 'shrink_conv'):
                module = getattr(self, name, None)
                if module is not None:
                    module.eval()
        return self

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

        if self.use_dir:
            for p in self.dir_head.parameters():
                p.requires_grad = False
    
    def regroup(self, x, record_len, k=1):
        cum_sum_len = torch.cumsum(record_len*k, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def select_current_frames(self, x, record_len, k=1):
        if k <= 1:
            return x
        current_frames = []
        for batch_x, cav_num in zip(self.regroup(x, record_len, k), record_len):
            cav_num = int(cav_num.item())
            frame_shape = batch_x.shape[1:]
            current_frames.append(batch_x.view(cav_num, k, *frame_shape)[:, 0])
        return torch.cat(current_frames, dim=0)

    def select_current_intervals(self, time_intervals, record_len, k=1):
        if k <= 1:
            return time_intervals
        current_intervals = []
        for batch_t, cav_num in zip(self.regroup(time_intervals.view(-1), record_len, k), record_len):
            cav_num = int(cav_num.item())
            current_intervals.append(batch_t.view(cav_num, k)[:, 0])
        return torch.cat(current_intervals, dim=0)

    def _generate_pred_bbx_frames_for_roi(self, dataset, *args, **kwargs):
        post_processor = getattr(dataset, 'post_processor', None)
        params = getattr(post_processor, 'params', {}) \
            if post_processor is not None else {}
        target_args = params.get('target_args', None)
        old_threshold = None
        if target_args is not None:
            old_threshold = target_args.get('score_threshold', None)

        def call_with_threshold(threshold):
            if threshold is not None and target_args is not None:
                target_args['score_threshold'] = float(threshold)
            return dataset.generate_pred_bbx_frames(*args, **kwargs)

        try:
            result = call_with_threshold(self.ectra_roi_score_threshold)
            fallback = self.ectra_roi_fallback_score_threshold
            if (fallback is not None and self.ectra_roi_min_count > 0 and
                    self._count_pred_bbx_frames(result) < self.ectra_roi_min_count):
                result = call_with_threshold(fallback)
                self._ectra_roi_threshold_fallback_count += 1
            return result
        finally:
            if old_threshold is not None:
                target_args['score_threshold'] = old_threshold

    @staticmethod
    def _count_pred_bbx_frames(box_results):
        count = 0
        for key, value in box_results.items():
            if not isinstance(key, int) or not isinstance(value, dict):
                continue
            boxes = value.get('pred_box_center_tensor', None)
            if torch.is_tensor(boxes):
                count += int(boxes.shape[0])
        return count

    def generate_box_flow(self, data_dict, pred_dict, dataset, device,
                          shape_list=None):
        if dataset is None:
            raise ValueError(
                'ECTRA ROI mode needs dataset.generate_pred_bbx_frames; '
                'run train.py/inference.py with --two_stage 1.')

        lidar_pose_batch = self.regroup(
            data_dict['past_lidar_pose'], data_dict['record_len'], k=1)
        past_k_time_diff = self.regroup(
            data_dict['past_k_time_interval'], data_dict['record_len'], k=self.k)
        anchor_box = data_dict['anchor_box']
        psm_single_list = pred_dict['psm_single_list']
        rm_single_list = pred_dict['rm_single_list']

        if shape_list is None:
            shape_list = torch.tensor([64, 200, 704], device=device)
        box_flow_map_list = []
        reserved_mask_list = []
        roi_context_list = []
        roi_mask_list = []
        roi_context_aux_list = []
        roi_diag_aux_list = []
        self._ectra_roi_threshold_fallback_count = 0
        roi_generation_calls = 0

        for b, lidar_pose in enumerate(lidar_pose_batch):
            cav_num = int(data_dict['record_len'][b].item())
            box_results = OrderedDict()
            psm_single = psm_single_list[b].reshape(
                cav_num, self.k, psm_single_list[b].shape[1],
                psm_single_list[b].shape[-2], psm_single_list[b].shape[-1])
            rm_single = rm_single_list[b].reshape(
                cav_num, self.k, rm_single_list[b].shape[1],
                rm_single_list[b].shape[-2], rm_single_list[b].shape[-1])
            cav_past_k_time_diff = past_k_time_diff[b].view(cav_num, self.k)
            single_score_stats = {}

            for cav_idx in range(cav_num):
                if self.diagnostic_roi_box_stats and dataset is not None:
                    post_processor = getattr(dataset, 'post_processor', None)
                    threshold = 0.0
                    if post_processor is not None:
                        threshold = post_processor.params['target_args'][
                            'score_threshold']
                    if self.ectra_roi_score_threshold is not None:
                        threshold = float(self.ectra_roi_score_threshold)
                    prob = torch.sigmoid(
                        psm_single[cav_idx].permute(0, 2, 3, 1))
                    prob = prob.reshape(self.k, -1)
                    single_score_stats[cav_idx] = {
                        'raw_anchor_count': (prob > threshold).float().sum(1),
                        'max_score': prob.max(1)[0],
                    }
                pastk_trans_mat = []
                for frame_idx in range(self.k):
                    unit_mat = x1_to_x2(
                        lidar_pose[cav_idx, frame_idx].cpu().numpy(),
                        lidar_pose[cav_idx, 0].cpu().numpy())
                    pastk_trans_mat.append(unit_mat)
                pastk_trans_mat = torch.from_numpy(
                    np.stack(pastk_trans_mat, axis=0)).to(device)

                try:
                    roi_generation_calls += 1
                    box_results[cav_idx] = self._generate_pred_bbx_frames_for_roi(
                        dataset,
                        psm_single[cav_idx],
                        rm_single[cav_idx],
                        pastk_trans_mat,
                        cav_past_k_time_diff[cav_idx],
                        anchor_box)
                except TypeError:
                    roi_generation_calls += 1
                    single_pred = {
                        'psm_single': psm_single[cav_idx],
                        'rm_single': rm_single[cav_idx],
                    }
                    box_results[cav_idx] = self._generate_pred_bbx_frames_for_roi(
                        dataset,
                        single_pred,
                        pastk_trans_mat,
                        cav_past_k_time_diff[cav_idx],
                        anchor_box)

            box_flow_map, reserved_mask = self.matcher(
                box_results, shape_list=shape_list, viz_flag=self.viz_bbx_flag)
            box_flow_map_list.append(box_flow_map)
            reserved_mask_list.append(reserved_mask)
            if self.ectra_roi_context_dim > 0:
                roi_context, batch_roi_masks, roi_context_aux = self.build_ectra_roi_context(
                    box_results, shape_list, device)
                if self.diagnostic_roi_box_stats:
                    roi_context_aux.update(self._diagnose_single_roi_predictions(
                        box_results, data_dict, b, lidar_pose, device,
                        single_score_stats))
                roi_context_list.append(roi_context)
                roi_mask_list.extend(batch_roi_masks)
                roi_context_aux_list.append(roi_context_aux)
            elif self.diagnostic_roi_box_stats:
                roi_diag_aux_list.append(self._diagnose_single_roi_predictions(
                    box_results, data_dict, b, lidar_pose, device,
                    single_score_stats))

        if self.ectra_roi_context_dim > 0 and roi_context_list:
            roi_context = {
                'dense': torch.cat(roi_context_list, dim=0),
                'roi_masks': roi_mask_list,
                'aux': self._merge_roi_context_aux(roi_context_aux_list),
            }
            roi_context['aux']['ectra_roi_threshold_fallback_mean'] = \
                torch.tensor(
                    float(self._ectra_roi_threshold_fallback_count) /
                    max(float(roi_generation_calls), 1.0),
                    device=device)
            if roi_diag_aux_list:
                roi_context['aux'].update(
                    self._merge_roi_context_aux(roi_diag_aux_list))
        elif self.diagnostic_roi_box_stats and roi_diag_aux_list:
            roi_context = {
                'aux': self._merge_roi_context_aux(roi_diag_aux_list),
            }
        else:
            roi_context = None
        return (torch.cat(box_flow_map_list, dim=0),
                torch.cat(reserved_mask_list, dim=0),
                roi_context)

    @staticmethod
    def _identity_grid(batch_size, height, width, device, dtype):
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * \
            (2.0 / height) - 1.0
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * \
            (2.0 / width) - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack((xx, yy), dim=-1)
        return grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def _empty_roi_context(self, shape_list, device, dtype=torch.float32):
        _, height, width = [int(x.item()) if torch.is_tensor(x) else int(x)
                           for x in shape_list]
        context = torch.zeros(
            self.ectra_roi_context_dim, height, width,
            device=device, dtype=dtype)
        masks = torch.zeros(0, 1, height, width, device=device, dtype=dtype)
        return context, masks

    @staticmethod
    def _safe_mean(values, device):
        if not values:
            return torch.tensor(0.0, device=device)
        values = [value.reshape(1).to(device=device, dtype=torch.float32)
                  for value in values]
        return torch.cat(values, dim=0).mean()

    def _merge_roi_context_aux(self, aux_list):
        if not aux_list:
            return {}
        device = next(iter(aux_list[0].values())).device
        merged = {}
        for key in aux_list[0].keys():
            values = [aux[key] for aux in aux_list if key in aux]
            merged[key] = self._safe_mean(values, device)
        return merged

    def _match_current_previous_rois(self, current, previous):
        current_centers = current['pred_box_center_tensor'][:, :2]
        previous_centers = previous['pred_box_center_tensor'][:, :2]
        if current_centers.shape[0] == 0 or previous_centers.shape[0] == 0:
            device = current_centers.device
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty, torch.zeros(0, device=device)

        cost = torch.cdist(previous_centers, current_centers)
        prev_np, curr_np = linear_sum_assignment(cost.detach().cpu().numpy())
        prev_ids = torch.as_tensor(
            prev_np, dtype=torch.long, device=current_centers.device)
        curr_ids = torch.as_tensor(
            curr_np, dtype=torch.long, device=current_centers.device)
        if prev_ids.numel() == 0:
            return prev_ids, curr_ids, torch.zeros(0, device=current_centers.device)

        match_dist = cost[prev_ids, curr_ids]
        keep = match_dist < self.ectra_roi_match_thresh
        return prev_ids[keep], curr_ids[keep], match_dist[keep]

    def _select_topk_rois(self, frame_result):
        if self.ectra_roi_context_topk <= 0:
            return frame_result
        scores = frame_result.get('scores', None)
        centers = frame_result.get('pred_box_center_tensor', None)
        if scores is None or centers is None:
            return frame_result
        if scores.numel() <= self.ectra_roi_context_topk:
            return frame_result
        keep = torch.topk(scores, self.ectra_roi_context_topk).indices
        selected = dict(frame_result)
        for key in ('scores', 'pred_box_center_tensor',
                    'pred_box_3dcorner_tensor', 'pred_box_2dcorner_tensor'):
            value = selected.get(key, None)
            if torch.is_tensor(value) and value.shape[0] == scores.shape[0]:
                selected[key] = value[keep]
        return selected

    def _rasterize_roi_context(self, context, roi_masks, centers, corners2d,
                               scores, velocity, displacement, match_valid,
                               match_dist, scale=2.5):
        if centers.shape[0] == 0:
            return
        _, height, width = context.shape
        dtype = context.dtype
        device = context.device

        corners2d = corners2d[:, :, :2].to(device=device, dtype=dtype)
        centers = centers.to(device=device, dtype=dtype)
        scores = scores.to(device=device, dtype=dtype).view(-1)
        velocity = velocity.to(device=device, dtype=dtype)
        displacement = displacement.to(device=device, dtype=dtype)
        match_valid = match_valid.to(device=device, dtype=dtype).view(-1)
        match_dist = match_dist.to(device=device, dtype=dtype).view(-1)

        warped = corners2d * scale + displacement[:, None, :] * scale
        x_min = (warped[:, :, 0].min(dim=1)[0] - 1 + int(width / 2)).long()
        x_max = (warped[:, :, 0].max(dim=1)[0] + 1 + int(width / 2)).long()
        y_min = (warped[:, :, 1].min(dim=1)[0] - 1 + int(height / 2)).long()
        y_max = (warped[:, :, 1].max(dim=1)[0] + 1 + int(height / 2)).long()
        x_min = torch.clamp(x_min, 0, width)
        x_max = torch.clamp(x_max, 0, width)
        y_min = torch.clamp(y_min, 0, height)
        y_max = torch.clamp(y_max, 0, height)

        cur_center = centers[:, :2] + displacement
        pos_scale_x, pos_scale_y = self.ectra_roi_position_scale
        yaw = centers[:, 6]
        values = torch.stack((
            torch.ones_like(scores),
            torch.clamp(cur_center[:, 0] / pos_scale_x, -2.0, 2.0),
            torch.clamp(cur_center[:, 1] / pos_scale_y, -2.0, 2.0),
            torch.clamp(centers[:, 3] / self.ectra_roi_size_scale, 0.0, 4.0),
            torch.clamp(centers[:, 4] / self.ectra_roi_size_scale, 0.0, 4.0),
            torch.clamp(centers[:, 5] / self.ectra_roi_size_scale, 0.0, 4.0),
            torch.sin(yaw),
            torch.cos(yaw),
            torch.clamp(velocity[:, 0] / self.ectra_roi_velocity_scale, -4.0, 4.0),
            torch.clamp(velocity[:, 1] / self.ectra_roi_velocity_scale, -4.0, 4.0),
            torch.clamp(displacement[:, 0] / self.ectra_roi_flow_scale, -4.0, 4.0),
            torch.clamp(displacement[:, 1] / self.ectra_roi_flow_scale, -4.0, 4.0),
            torch.clamp(scores, 0.0, 1.0),
            match_valid,
            torch.clamp(match_dist / max(self.ectra_roi_match_thresh, 1e-3), 0.0, 4.0),
        ), dim=1)

        for roi_idx in range(centers.shape[0]):
            if x_max[roi_idx] <= x_min[roi_idx] or y_max[roi_idx] <= y_min[roi_idx]:
                continue
            y0, y1 = y_min[roi_idx].item(), y_max[roi_idx].item()
            x0, x1 = x_min[roi_idx].item(), x_max[roi_idx].item()
            used_channels = min(self.ectra_roi_context_dim, values.shape[1])
            context[:used_channels, y0:y1, x0:x1] = \
                values[roi_idx, :used_channels].view(-1, 1, 1)
            roi_masks.append(context.new_zeros(1, height, width))
            roi_masks[-1][:, y0:y1, x0:x1] = 1.0

    def _gt_boxes_for_batch(self, data_dict, batch_idx, device):
        centers = data_dict.get('object_bbx_center', None)
        masks = data_dict.get('object_bbx_mask', None)
        if centers is None or masks is None:
            return torch.zeros(0, 7, device=device)

        centers = centers.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device)
        if centers.dim() == 2:
            batch_centers = centers
            batch_masks = masks
        else:
            batch_centers = centers[batch_idx]
            batch_masks = masks[batch_idx]

        valid = batch_masks > 0
        if valid.numel() == 0:
            return torch.zeros(0, 7, device=device)
        return batch_centers[valid]

    def _center_match_summary(self, pred_centers, gt_centers, device):
        pred_count = torch.tensor(float(pred_centers.shape[0]), device=device)
        gt_count = torch.tensor(float(gt_centers.shape[0]), device=device)
        if pred_centers.shape[0] == 0 or gt_centers.shape[0] == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                'pred_count': pred_count,
                'gt_count': gt_count,
                'gt_recall': zero,
                'pred_precision': zero,
                'min_center_dist': zero,
            }

        dist = torch.cdist(gt_centers[:, :2], pred_centers[:, :2])
        gt_min = dist.min(dim=1)[0]
        pred_min = dist.min(dim=0)[0]
        thresh = self.diagnostic_roi_match_distance
        return {
            'pred_count': pred_count,
            'gt_count': gt_count,
            'gt_recall': (gt_min < thresh).float().mean(),
            'pred_precision': (pred_min < thresh).float().mean(),
            'min_center_dist': gt_min.mean(),
        }

    @staticmethod
    def _batch_anchor_box(anchor_box, batch_idx):
        if torch.is_tensor(anchor_box) and anchor_box.dim() == 5:
            return anchor_box[batch_idx]
        return anchor_box

    def _decode_head_centers_for_diagnostics(self, dataset, psm, rm, anchor_box,
                                             batch_idx):
        device = psm.device
        post_processor = getattr(dataset, 'post_processor', None)
        if post_processor is None:
            return torch.zeros(0, 7, device=device), torch.tensor(0.0, device=device)

        threshold = post_processor.params['target_args']['score_threshold']
        prob = torch.sigmoid(psm[batch_idx:batch_idx + 1].permute(0, 2, 3, 1))
        prob = prob.reshape(1, -1)
        raw_count = (prob[0] > threshold).float().sum()
        if raw_count.item() == 0:
            return torch.zeros(0, 7, device=device), raw_count

        anchors = self._batch_anchor_box(anchor_box, batch_idx)
        batch_box3d = post_processor.delta_to_boxes3d(
            rm[batch_idx:batch_idx + 1], anchors)
        mask = torch.gt(prob, threshold).view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
        boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
        scores = torch.masked_select(prob[0], mask[0])
        if boxes3d.shape[0] == 0:
            return torch.zeros(0, 7, device=device), raw_count

        boxes3d_corner = box_utils.boxes_to_corners_3d(
            boxes3d, order=post_processor.params['order'])
        keep_index_1 = box_utils.remove_large_pred_bbx(boxes3d_corner)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(boxes3d_corner)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)
        boxes3d_corner = boxes3d_corner[keep_index]
        scores = scores[keep_index]
        if boxes3d_corner.shape[0] == 0:
            return torch.zeros(0, 7, device=device), raw_count

        keep_index = box_utils.nms_rotated(
            boxes3d_corner, scores, post_processor.params['nms_thresh'])
        boxes3d_corner = boxes3d_corner[keep_index]
        if boxes3d_corner.shape[0] == 0:
            return torch.zeros(0, 7, device=device), raw_count

        range_mask = box_utils.get_mask_for_boxes_within_range_torch(
            boxes3d_corner, post_processor.params['gt_range'])
        boxes3d_corner = boxes3d_corner[range_mask]
        if boxes3d_corner.shape[0] == 0:
            return torch.zeros(0, 7, device=device), raw_count
        return box_utils.corner_to_center_torch(
            boxes3d_corner, post_processor.params['order']), raw_count

    def _project_roi_centers_to_ego(self, frame_result, transform, device):
        corners = frame_result.get('pred_box_3dcorner_tensor', None)
        if not torch.is_tensor(corners) or corners.shape[0] == 0:
            return torch.zeros(0, 7, device=device)
        transform = torch.as_tensor(transform, device=device, dtype=torch.float32)
        projected = box_utils.project_box3d(corners.to(device), transform)
        return box_utils.corner_to_center_torch(projected, order='hwl')

    def _diagnose_single_roi_predictions(self, box_results, data_dict,
                                         batch_idx, lidar_pose, device,
                                         single_score_stats=None):
        gt_centers = self._gt_boxes_for_batch(data_dict, batch_idx, device)
        pred_centers = []
        post_frame_counts = []
        raw_anchor_counts = []
        max_scores = []
        ego_pose = lidar_pose[0, 0].detach().cpu().numpy()
        for cav_idx, cav_content in box_results.items():
            if cav_idx == 'past_k_time_diff' or 0 not in cav_content:
                continue
            if single_score_stats and cav_idx in single_score_stats:
                raw_anchor_counts.extend([
                    value.detach()
                    for value in single_score_stats[cav_idx]['raw_anchor_count']
                ])
                max_scores.extend([
                    value.detach()
                    for value in single_score_stats[cav_idx]['max_score']
                ])
            for frame_idx, frame_result in cav_content.items():
                if not isinstance(frame_idx, int):
                    continue
                centers = frame_result.get('pred_box_center_tensor', None)
                count = float(centers.shape[0]) if torch.is_tensor(centers) else 0.0
                post_frame_counts.append(torch.tensor(count, device=device))
                if frame_idx == 0 and torch.is_tensor(centers):
                    cav_pose = lidar_pose[cav_idx, 0].detach().cpu().numpy()
                    transform = x1_to_x2(cav_pose, ego_pose)
                    pred_centers.append(self._project_roi_centers_to_ego(
                        frame_result, transform, device))

        if pred_centers:
            pred_centers = torch.cat(pred_centers, dim=0)
        else:
            pred_centers = torch.zeros(0, 7, device=device)
        stats = self._center_match_summary(pred_centers, gt_centers, device)
        return {
            'ectra_diag_single_roi_boxes_mean': stats['pred_count'],
            'ectra_diag_gt_centered_roi_count_mean': stats['gt_count'],
            'ectra_diag_single_roi_gt_recall': stats['gt_recall'],
            'ectra_diag_single_roi_pred_precision': stats['pred_precision'],
            'ectra_diag_single_roi_min_center_dist': stats['min_center_dist'],
            'ectra_diag_single_roi_frame_boxes_mean': self._safe_mean(
                post_frame_counts, device),
            'ectra_diag_single_roi_raw_anchor_mean': self._safe_mean(
                raw_anchor_counts, device),
            'ectra_diag_single_roi_max_score_mean': self._safe_mean(
                max_scores, device),
        }

    def _diagnose_fused_predictions(self, data_dict, dataset, psm, rm):
        device = psm.device
        batch_stats = []
        raw_counts = []
        batch_size = psm.shape[0]
        for batch_idx in range(batch_size):
            gt_centers = self._gt_boxes_for_batch(data_dict, batch_idx, device)
            pred_centers, raw_count = self._decode_head_centers_for_diagnostics(
                dataset, psm, rm, data_dict['anchor_box'], batch_idx)
            raw_counts.append(raw_count.detach())
            batch_stats.append(self._center_match_summary(
                pred_centers.detach(), gt_centers.detach(), device))
        return {
            'ectra_diag_final_fused_boxes_mean': self._safe_mean(
                [x['pred_count'] for x in batch_stats], device),
            'ectra_diag_final_fused_raw_anchor_mean': self._safe_mean(
                raw_counts, device),
            'ectra_diag_final_fused_gt_recall': self._safe_mean(
                [x['gt_recall'] for x in batch_stats], device),
            'ectra_diag_final_fused_pred_precision': self._safe_mean(
                [x['pred_precision'] for x in batch_stats], device),
            'ectra_diag_final_fused_min_center_dist': self._safe_mean(
                [x['min_center_dist'] for x in batch_stats], device),
        }

    def build_ectra_roi_context(self, box_results, shape_list, device):
        """
        Convert per-frame ROI detections into dense RNN context maps.

        Channels are, in order: ROI mask, normalized center x/y, box size
        h/w/l, sin/cos yaw, estimated velocity x/y, extrapolated displacement
        x/y, score, match-valid flag, and normalized match distance.
        """
        dense_context = []
        roi_masks_by_cav = []
        raw_roi_counts = []
        total_roi_counts = []
        matched_roi_counts = []
        unmatched_roi_counts = []
        match_dist_means = []
        velocity_abs_means = []
        displacement_abs_means = []
        score_means = []
        for cav_idx, cav_content in box_results.items():
            if cav_idx == 'past_k_time_diff':
                continue
            context, masks = self._empty_roi_context(shape_list, device)
            roi_masks = []
            if cav_idx != 0 and 0 in cav_content and 1 in cav_content:
                current_raw = cav_content[0]
                previous_raw = cav_content[1]
                current = self._select_topk_rois(current_raw)
                previous = self._select_topk_rois(previous_raw)
                prev_ids, curr_ids, match_dist = self._match_current_previous_rois(
                    current, previous)

                current_centers = current['pred_box_center_tensor']
                current_scores = current['scores']
                raw_roi_counts.append(torch.tensor(
                    float(current_raw['pred_box_center_tensor'].shape[0]),
                    device=device))
                total_roi_counts.append(torch.tensor(
                    float(current_centers.shape[0]), device=device))
                matched_roi_counts.append(torch.tensor(
                    float(curr_ids.numel()), device=device))
                unmatched_roi_counts.append(torch.tensor(
                    float(max(current_centers.shape[0] - curr_ids.numel(), 0)),
                    device=device))
                if curr_ids.numel() > 0:
                    t0 = torch.as_tensor(
                        cav_content['past_k_time_diff'][0],
                        device=device, dtype=current_centers.dtype)
                    t1 = torch.as_tensor(
                        cav_content['past_k_time_diff'][1],
                        device=device, dtype=current_centers.dtype)
                    dt = t0 - t1
                    dt = torch.where(
                        torch.abs(dt) < 1e-3, torch.ones_like(dt), dt)
                    matched_current = current_centers[curr_ids]
                    matched_previous = previous['pred_box_center_tensor'][prev_ids]
                    velocity = (matched_current[:, :2] -
                                matched_previous[:, :2]) / dt
                    displacement = velocity * (0 - t0)
                    score = 0.5 * (
                        current_scores[curr_ids] + previous['scores'][prev_ids])
                    match_dist_means.append(match_dist.detach().float().mean())
                    velocity_abs_means.append(
                        velocity.detach().float().abs().mean())
                    displacement_abs_means.append(
                        displacement.detach().float().abs().mean())
                    score_means.append(score.detach().float().mean())
                    corners2d = box_utils.boxes_to_corners2d(
                        matched_current, order='hwl')
                    self._rasterize_roi_context(
                        context, roi_masks, matched_current, corners2d, score,
                        velocity, displacement,
                        torch.ones_like(score), match_dist)

                if current_centers.shape[0] > curr_ids.numel():
                    matched = torch.zeros(
                        current_centers.shape[0], dtype=torch.bool,
                        device=current_centers.device)
                    if curr_ids.numel() > 0:
                        matched[curr_ids] = True
                    remain_ids = torch.where(~matched)[0]
                    if remain_ids.numel() > 0:
                        remain_centers = current_centers[remain_ids]
                        remain_scores = current_scores[remain_ids]
                        score_means.append(remain_scores.detach().float().mean())
                        zeros = torch.zeros(
                            remain_ids.numel(), 2,
                            device=device, dtype=remain_centers.dtype)
                        corners2d = box_utils.boxes_to_corners2d(
                            remain_centers, order='hwl')
                        self._rasterize_roi_context(
                            context, roi_masks, remain_centers, corners2d,
                            remain_scores, zeros, zeros,
                            torch.zeros_like(remain_scores),
                            torch.ones_like(remain_scores) *
                            self.ectra_roi_match_thresh)

            if roi_masks:
                masks = torch.stack(roi_masks, dim=0)
            dense_context.append(context.unsqueeze(0))
            roi_masks_by_cav.append(masks)
        total = self._safe_mean(total_roi_counts, device)
        matched = self._safe_mean(matched_roi_counts, device)
        aux = {
            'ectra_roi_raw_total_mean': self._safe_mean(
                raw_roi_counts, device),
            'ectra_roi_total_mean': total,
            'ectra_roi_matched_mean': matched,
            'ectra_roi_unmatched_mean': self._safe_mean(
                unmatched_roi_counts, device),
            'ectra_roi_match_ratio': matched / total.clamp_min(1.0),
            'ectra_roi_match_dist_mean': self._safe_mean(
                match_dist_means, device),
            'ectra_roi_velocity_abs_mean': self._safe_mean(
                velocity_abs_means, device),
            'ectra_roi_displacement_abs_mean': self._safe_mean(
                displacement_abs_means, device),
            'ectra_roi_score_mean': self._safe_mean(score_means, device),
        }
        return torch.cat(dense_context, dim=0), roi_masks_by_cav, aux

    def forward(self, data_dict, dataset=None):
        set_decoder_domain(self, False, self.training)
        voxel_features = data_dict['processed_lidar']['voxel_features']         #(M, 32, 4)
        voxel_coords = data_dict['processed_lidar']['voxel_coords']             #(M, 4)
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']     #(M, )
        record_len = data_dict['record_len']
        record_frames = data_dict['past_k_time_interval']                       #(B, )
        pairwise_t_matrix = data_dict['pairwise_t_matrix']                      #(B, L, k, 4, 4)
        
        # debug = 0
        B, _, k, _, _ = pairwise_t_matrix.shape
        # for i in range(B):
        #     debug += record_len[i]*k

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c  ('pillar_features')
        batch_dict = self.pillar_vfe(batch_dict)
        # (n, c) -> (batch_cav_size, C, H, W) put pillars into spatial feature map ('spatial_features')
        # import ipdb; ipdb.set_trace()
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict) # 'spatial_features_2d': (batch_cav_size, 128*3, H/2, W/2)
        flow_shape_list = torch.tensor(
            batch_dict['spatial_features'].shape[-3:],
            device=batch_dict['spatial_features'].device)
        ectra_aux = {}
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d']

        noise_pairwise_t_matrix = None
        if self.noise_flag and B==1:
            # noise_level = {'pos_std': 0.5, 'rot_std': 0, 'pos_mean': 0, 'rot_mean': 0}
            noise_pairwise_t_matrix = get_past_k_pairwise_transformation2ego(data_dict['past_lidar_pose'], self.noise_level, k=self.k, max_cav=5)[:record_len[0]].unsqueeze(0)
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)  # (B, 256, H', W')
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        ####### debug use, viz feature of each cav
        # from matplotlib import pyplot as plt
        # viz_save_path = '/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/debug_4_feature_flow/where2comm'
        # for i in range(spatial_features_2d.shape[0]):
        #     viz_content = torch.max(spatial_features_2d[i], dim=0)[0].detach().cpu()
        #     plt.imshow(viz_content)
        #     plt.savefig(viz_save_path+f'/updated_feature_{i}.png')
        ##############
        
        # dcn
        # if self.dcn:
        #     spatial_features_2d = self.dcn_net(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        if self.use_dir:
            dm_single = self.dir_head(spatial_features_2d)
        psm_fusion = self.select_current_frames(psm_single, record_len, k)
        set_decoder_domain(self, True, self.training)
        roi_aux = {}
        box_flow_map = None
        reserved_mask = None
        roi_context = None
        flow_gt = data_dict['label_dict'].get('flow_gt', None) \
            if 'label_dict' in data_dict else None

        if self.ectra_roi_flag:
            psm_for_box = psm_single.detach()
            rm_for_box = rm_single.detach()
            single_output = {
                'psm_single_list': self.regroup(psm_for_box, record_len, k),
                'rm_single_list': self.regroup(rm_for_box, record_len, k),
            }
            box_flow_map, reserved_mask, roi_context = self.generate_box_flow(
                data_dict, single_output, dataset, psm_single.device,
                shape_list=flow_shape_list)

        if self.ectra_dense_enabled:
            batch_dict['spatial_features'], ectra_aux = self.ectra(
                batch_dict['spatial_features'], record_len, record_frames,
                roi_context=roi_context)
            if isinstance(roi_context, dict):
                ectra_aux.update(roi_context.get('aux', {}))

        fusion_spatial_features = self.select_current_frames(
            batch_dict['spatial_features'], record_len, k)
        fusion_pairwise_t_matrix = pairwise_t_matrix[:, :, 0:1]
        fusion_record_frames = self.select_current_intervals(
            record_frames, record_len, k)

        if self.ectra_roi_flag:
            box_flow_map, reserved_mask, roi_aux = self.ectra_roi(
                box_flow_map, reserved_mask, fusion_spatial_features,
                record_len, fusion_record_frames, flow_gt=flow_gt)

        # rain attention:
        fusion_backbone = self.fused_backbone if self.independent_fusion_backbone else self.backbone
        if self.multi_scale:
            if self.ectra_roi_flag:
                fused_feature, communication_rates, result_dict = self.rain_fusion(
                    batch_dict['spatial_features'],
                    psm_single,
                    record_len,
                    pairwise_t_matrix,
                    record_frames,
                    fusion_backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head],
                    box_flow=box_flow_map,
                    reserved_mask=reserved_mask,
                    flow_gt=flow_gt,
                    noise_pairwise_t_matrix=noise_pairwise_t_matrix,
                    num_roi_thres=self.num_roi_thres)
            elif self.compensation:
                if self.single_supervise:
                    fused_feature, single_feature, communication_rates, all_recon_loss, result_dict = self.rain_fusion(fusion_spatial_features,
                                                psm_fusion,
                                                record_len,
                                                fusion_pairwise_t_matrix, 
                                                fusion_record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])
                else:
                    fused_feature,fused_feature_curr,fused_feature_latency, communication_rates, all_recon_loss, all_latency_recon_loss, result_dict = self.rain_fusion(fusion_spatial_features,
                                                psm_fusion,
                                                record_len,
                                                fusion_pairwise_t_matrix, 
                                                fusion_record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])
            else: 
                fused_feature, communication_rates, result_dict = self.rain_fusion(fusion_spatial_features,
                    psm_fusion,
                    record_len,
                    fusion_pairwise_t_matrix, 
                    fusion_record_frames,
                    self.backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head],
                    noise_pairwise_t_matrix=noise_pairwise_t_matrix)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fusion_shrink = self.fused_shrink_conv if self.independent_fusion_backbone else self.shrink_conv
                fused_feature = fusion_shrink(fused_feature)
                if self.single_supervise:
                    single_feature = self.shrink_conv(single_feature)
                # if self.compensation:
                    
                #         # single_feature = self.shrink_conv(single_feature)
                
                #         fused_feature_curr = self.shrink_conv(fused_feature_curr)
                #         fused_feature_latency = self.shrink_conv(fused_feature_latency)
        else:
            fused_feature, communication_rates, result_dict = self.rain_fusion(spatial_features_2d,
                                            psm_fusion,
                                            record_len,
                                            fusion_pairwise_t_matrix,
                                            fusion_record_frames)
            if self.compensation:
                if self.single_supervise:
                    fused_feature, single_feature, communication_rates, all_recon_loss, result_dict = self.rain_fusion(spatial_features_2d,
                                                psm_fusion,
                                                record_len,
                                                fusion_pairwise_t_matrix, 
                                                fusion_record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])
                else:
                    fused_feature,fused_feature_curr,fused_feature_latency, communication_rates, all_recon_loss, all_latency_recon_loss, result_dict = self.rain_fusion(spatial_features_2d,
                                                psm_fusion,
                                                record_len,
                                                fusion_pairwise_t_matrix, 
                                                fusion_record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])            

        ####### debug use, viz fused feature of where2comm
        # viz_content = torch.max(fused_feature[0], dim=0)[0].detach().cpu()
        # plt.imshow(viz_content)
        # plt.savefig(viz_save_path+'/fused_feature.png')  
        ##############
        
        # print('fused_feature: ', fused_feature.shape)
        psm = self.fused_cls_head(fused_feature)
        rm = self.fused_reg_head(fused_feature)
        
        # fuse 之后的 feature (ego)
        output_dict = {'psm': psm,
                       'rm': rm}
        
        if self.use_dir:
            dm = self.fused_dir_head(fused_feature)
            output_dict.update({'dm': dm})

        if self.diagnostic_roi_box_stats and dataset is not None:
            with torch.no_grad():
                output_dict.update(self._diagnose_fused_predictions(
                    data_dict, dataset, psm.detach(), rm.detach()))

        if self.compensation:
            if self.single_supervise:
                psm_nonego_single = self.cls_head(single_feature)
                rm_nonego_single = self.reg_head(single_feature)
                output_dict.update({
                    'psm_nonego_single': psm_nonego_single,
                    'rm_nonego_single': rm_nonego_single
                })
            
            output_dict.update({
                'recon_loss': all_recon_loss, 
                'record_len': record_len
            })

        output_dict.update({'psm_single': psm_single,
                       'rm_single': rm_single,
                       'comm_rate': communication_rates
                       })
        if self.use_dir:
            output_dict.update({'dm_single': dm_single})
        
        output_dict.update(result_dict) 
        output_dict.update(ectra_aux)
        output_dict.update(roi_aux)
        
        return output_dict
