from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.codyntrust_base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv, AttentionDownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.fuse_modules.raindrop_swin import raindrop_swin
from opencood.models.fuse_modules.raindrop_swin_w_single import raindrop_swin_w_single
from opencood.models.fuse_modules.raindrop_flow import raindrop_fuse
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world
from opencood.tools.matcher import Matcher
from collections import OrderedDict
import torch
import numpy as np
from opencood.utils import box_utils

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


class PointPillarCodyntrustCobevflowBaseline(nn.Module):
    def __init__(self, args):
        super(PointPillarCodyntrustCobevflowBaseline, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # self.backbone = MaxResNetBEVBackbone(args['base_bev_backbone'], 64)
        if 'resnet' in args['base_bev_backbone'] and args['base_bev_backbone']['resnet']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
            self.fused_backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)            
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
    
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            print("===use downsample conv to reduce memory===")
            self.shrink_flag = True
            if args['shrink_header']['use_atten']:
                print("  ===AttentionDownsampleConv===")
                self.shrink_conv = AttentionDownsampleConv(args['shrink_header'])
                self.fused_shrink_conv = AttentionDownsampleConv(args['shrink_header'])
            else:
                self.shrink_conv = DownsampleConv(args['shrink_header'])
                self.fused_shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        # if args['compression'] > 0:
        #     self.compression = True
        #     self.naive_compressor = NaiveCompressor(256, args['compression'])

        if 'num_sweep_frames' in args:    # number of frames we use in LSTM
            self.k = args['num_sweep_frames']
        else:
            self.k = 0

        if 'time_delay' in args:          # number of time delay
            self.tau = args['time_delay'] 
        else:
            self.tau = 0

        self.dcn = False
        # if 'dcn' in args:
        #     self.dcn = True
        #     self.dcn_net = DCNNet(args['dcn'])

        self.design_mode = 0
        if 'design_mode' in args.keys():
            self.design_mode = args['design_mode']
            print(f'=== design mode : {self.design_mode} ===')

        self.noise_flag = 0
        if 'noise' in args.keys():
            self.noise_flag = True
            self.noise_level = {'pos_std': args['noise']['pos_std'], 'rot_std': args['noise']['rot_std'], 'pos_mean': args['noise']['pos_mean'], 'rot_mean': args['noise']['rot_mean']}
            print(f'=== noise level : {self.noise_level} ===')

        self.num_roi_thres = -1
        if 'num_roi_thres' in args.keys():
            self.num_roi_thres = args['num_roi_thres']
            print(f'=== num_roi_thres : {self.num_roi_thres} ===')
        self.roi_score_threshold = args.get('roi_score_threshold', None)
        diagnostics_args = args.get('diagnostics', {})
        self.diagnostic_roi_box_stats = bool(
            diagnostics_args.get('roi_box_stats', False))
        self.diagnostic_roi_match_distance = float(
            diagnostics_args.get('roi_match_distance', 4.0))

        self.single_supervise = False
        if 'with_compensation' in args and args['with_compensation']: # 如果已经有补偿
            self.compensation = True
            if 'with_single_supervise' in args and args['with_single_supervise']: # 单车监督，ROI生成？
                self.rain_fusion = raindrop_swin_w_single(args['rain_model'])
                self.single_supervise = True
            else: # 最终的融合检测器
                self.rain_fusion = raindrop_swin(args['rain_model'])
        else: 
            self.compensation = False
            self.rain_fusion = raindrop_fuse(args['rain_model'], self.design_mode) # 训练流

        self.multi_scale = args['rain_model']['multi_scale']
        
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
            self.fused_dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'], # num_bins 设置成多种桶
                                    kernel_size=1)
        else:
            self.use_dir = False

        if self.design_mode == 0:
            self.matcher = Matcher('flow')
        else:
            self.matcher = Matcher('linear')

        self.backbone_fix_flag = False
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix_flag = True
            self.backbone_fix()
            print('=== backbone fixed ===')

        self.only_tune_header_flag = False
        if 'only_tune_header' in args.keys() and args['only_tune_header']:
            self.only_tune_header_flag = True
            self.only_tune_header()
            print('=== only tune header ===')

        self.viz_bbx_flag = False
        if 'viz_bbx_flag' in args.keys() and args['viz_bbx_flag']:
            self.viz_bbx_flag = True
        
        assert self.backbone_fix_flag == False or self.only_tune_header_flag == False, 'backbone_fix and only_tune_header cannot be True at the same time'
    
    def only_tune_header(self):
        """
        Fix the parameters of backbone during finetune on timedelay
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.rain_fusion.parameters():
            p.requires_grad = False

        for p in self.matcher.parameters():
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
        '''
        x的形状为(B, C, H, W)
        record_len的形状为(B) 记录每一个样本场景下的agent个数
        k为保存的帧数

        '''
        cum_sum_len = torch.cumsum(record_len*k, dim=0) # 求累计和
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu()) # 分割批数据，返回的是List
        return split_x # List[p1()]

    def bandwidth_filter(self, input, num_box):
        topk_idx = torch.argsort(input['scores'], descending=True)[:num_box]

        output = {}
        output['scores'] = input['scores'][topk_idx]
        output['pred_box_3dcorner_tensor']  = input['pred_box_3dcorner_tensor'][topk_idx]
        output['pred_box_center_tensor'] = input['pred_box_center_tensor'][topk_idx]

        return output

    @staticmethod
    def _safe_mean(values, device):
        if not values:
            return torch.tensor(0.0, device=device)
        values = [
            value.reshape(1).to(device=device, dtype=torch.float32)
            for value in values
        ]
        return torch.cat(values, dim=0).mean()

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

    def _merge_box_diag_aux(self, aux_list, device):
        if not aux_list:
            return {}
        keys = sorted({key for aux in aux_list for key in aux.keys()})
        return {
            key: self._safe_mean([aux[key] for aux in aux_list if key in aux],
                                 device)
            for key in keys
        }

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
            'cobevflow_diag_single_roi_boxes_mean': stats['pred_count'],
            'cobevflow_diag_gt_centered_roi_count_mean': stats['gt_count'],
            'cobevflow_diag_single_roi_gt_recall': stats['gt_recall'],
            'cobevflow_diag_single_roi_pred_precision': stats['pred_precision'],
            'cobevflow_diag_single_roi_min_center_dist': stats['min_center_dist'],
            'cobevflow_diag_single_roi_frame_boxes_mean': self._safe_mean(
                post_frame_counts, device),
            'cobevflow_diag_single_roi_raw_anchor_mean': self._safe_mean(
                raw_anchor_counts, device),
            'cobevflow_diag_single_roi_max_score_mean': self._safe_mean(
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
            'cobevflow_diag_final_fused_boxes_mean': self._safe_mean(
                [x['pred_count'] for x in batch_stats], device),
            'cobevflow_diag_final_fused_raw_anchor_mean': self._safe_mean(
                raw_counts, device),
            'cobevflow_diag_final_fused_gt_recall': self._safe_mean(
                [x['gt_recall'] for x in batch_stats], device),
            'cobevflow_diag_final_fused_pred_precision': self._safe_mean(
                [x['pred_precision'] for x in batch_stats], device),
            'cobevflow_diag_final_fused_min_center_dist': self._safe_mean(
                [x['min_center_dist'] for x in batch_stats], device),
        }

    def _generate_pred_bbx_frames_for_roi(self, dataset, *args, **kwargs):
        post_processor = getattr(dataset, 'post_processor', None)
        params = getattr(post_processor, 'params', {}) \
            if post_processor is not None else {}
        target_args = params.get('target_args', None)
        old_threshold = None
        if target_args is not None:
            old_threshold = target_args.get('score_threshold', None)
            if self.roi_score_threshold is not None:
                target_args['score_threshold'] = float(
                    self.roi_score_threshold)
        try:
            return dataset.generate_pred_bbx_frames(*args, **kwargs)
        finally:
            if old_threshold is not None:
                target_args['score_threshold'] = old_threshold

    def generate_box_flow(self, data_dict, pred_dict, dataset, shape_list, device): 
        """
        data_dict : 
        pred_dict: 单车检测结果
        dataset: 数据集对象
        pred_dict : 
            {
                psm_single_list: len = B, each element's shape is (N_b x k, 2, H, W) 
                rm_single_list: len = B, each element's shape is (N_b x k, 14, H, W) 
            }
        """
        # for b in range(B):
        # 1. get box results
        lidar_pose_batch = self.regroup(data_dict['past_lidar_pose'], data_dict['record_len'], k=1) # lidar pose一般是6dof，猜测形状为(B, k, 6) 通过record_len划分为List 每个元素为(N_b, k, 6)
        past_k_time_diff = self.regroup(data_dict['past_k_time_interval'], data_dict['record_len'], k=self.k) # k帧的时间间隔 长度为 (B) 返回List 每个元素为(N_b x k)  其实这里就是(N_b x 3)
        anchor_box = data_dict['anchor_box'] # (H, W, 2, 7) 这是预设置的锚框
        psm_single_list = pred_dict['psm_single_list'] # (N_b x k, 2, H, W)  列表每个元素都是前面这个形状，列表的长度为B
        rm_single_list = pred_dict['rm_single_list'] # (N_b x k, 2*7, H, W)
        dm_single_list = pred_dict['dm_single_list'] # 方向预测 (N_b x k, 2*2, H, W)

        # H, W = psm_single_list[0].shape[-2:]
        # shape_list = torch.tensor([64, H, W]).to(device)
        
        trans_mat_pastk_2_past0_batch = []
        B = len(lidar_pose_batch)
        box_flow_map_list = []
        reserved_mask_list = []
        diag_aux_list = []
        self._last_roi_diagnostics = {}

        if self.viz_bbx_flag:
            ori_reserved_mask_list = []
            single_box_results = None
        
        # for all batches
        for b in range(B):
            box_results = OrderedDict()
            psm_single = psm_single_list[b].reshape(-1, self.k, 2, psm_single_list[b].shape[-2], psm_single_list[b].shape[-1]) # (N_b, k, 2, H, W)
            rm_single = rm_single_list[b].reshape(-1, self.k, 14, rm_single_list[b].shape[-2], rm_single_list[b].shape[-1]) # (N_b, k, 14, H, W)
            if self.use_dir:
                dm_single = dm_single_list[b].reshape(-1, self.k, 4, dm_single_list[b].shape[-2], dm_single_list[b].shape[-1]) # (N_b, k, 4, H, W)
            single_score_stats = {}

            cav_past_k_time_diff = past_k_time_diff[b] # (N_b x k)  从列表中选择出一个样本场景，这表示了每一帧到第0帧的时间间隔  而且注意上面在regroup时，传递的k一个是1一个是3，那是因为一个的形状为(B, k, 6)，而data_dict['past_k_time_interval']的形状为B，所以他需要额外按照k的大小来继续划分，这里就看出来k在代码中默认是3，也就是每个车存储了3帧
            cav_trans_mat_pastk_2_past0 = []
            # for all cavs
            '''
            box_result : dict for each cav at each time
            {
                cav_idx : {
                    'past_k_time_diff' : 
                    [0] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. (n, 8, 3)
                        pred_box_center_tensor : (n, 7)
                        scores: (n, )
                    },
                    ...
                    [k-1] : { ... }
                }
            }
            '''
            comm_volum = []
            for cav_idx in range(data_dict['record_len'][b]):# 遍历一个场景下的所有车 循环次数是该场景下的cav数
                if self.diagnostic_roi_box_stats and dataset is not None:
                    post_processor = getattr(dataset, 'post_processor', None)
                    threshold = 0.0
                    if post_processor is not None:
                        threshold = post_processor.params['target_args'][
                            'score_threshold']
                    if self.roi_score_threshold is not None:
                        threshold = float(self.roi_score_threshold)
                    prob = torch.sigmoid(
                        psm_single[cav_idx].permute(0, 2, 3, 1))
                    prob = prob.reshape(self.k, -1)
                    single_score_stats[cav_idx] = {
                        'raw_anchor_count': (prob > threshold).float().sum(1),
                        'max_score': prob.max(1)[0],
                    }
                # generate one cav's trans_mat_pastk_2_past0
                pastk_trans_mat_pastk_2_past0 = []
                for i in range(self.k): # 遍历k帧
                    unit_mat = x1_to_x2(lidar_pose_batch[b][cav_idx, i, :].cpu().numpy(), lidar_pose_batch[b][cav_idx, 0].cpu()) # (4, 4) 一个agent的第i帧位置到第0帧的变换矩阵
                    pastk_trans_mat_pastk_2_past0.append(unit_mat)# 存放每一帧到第0帧的变换矩阵， 所以一共有k个元素
                pastk_trans_mat_pastk_2_past0 = torch.from_numpy(np.stack(pastk_trans_mat_pastk_2_past0, axis=0)).to(device) # (k, 4, 4)

                m_single = {}
                m_single['psm_single'] = psm_single[cav_idx] # (k, 2, H, W) 每一个agent的
                m_single['rm_single'] = rm_single[cav_idx] # (k, 14, H, W)
                if self.use_dir:
                    m_single['dm_single'] = dm_single[cav_idx] #(k, 4, H, W)
                # 1. generate one cav's box results 接下来这一步本质上是过滤，选出合适的bbx作为预测结果 输入k帧的检测结果
                box_results[cav_idx] = self._generate_pred_bbx_frames_for_roi(
                    dataset, m_single, pastk_trans_mat_pastk_2_past0,
                    cav_past_k_time_diff[cav_idx*self.k:cav_idx*self.k+self.k],
                    anchor_box)



            comm_rois_nums = box_results[0][0]['scores'].shape[0] # 只计算ego向外发送的
            comm_volum.append(comm_rois_nums * 40)

            cav_trans_mat_pastk_2_past0.append(pastk_trans_mat_pastk_2_past0)
            
            # 2. generate box flow in one batch
            if self.viz_bbx_flag: # 这个应该是判断是否需要可视化单车检测的bbx流数据 额外返回了 ori_mask：（N_b， C， H，W）标记了object的位置，置1 matched_idx_list：列表 其中元素有 (N_obj, 3)  也有 (N_obj, 2) compensated_results_list: 列表 (N_obj, 4, 2)
                box_flow_map, mask, ori_mask, matched_idx_list, compensated_results_list = self.matcher(box_results, shape_list=shape_list, viz_flag=self.viz_bbx_flag)
                ori_reserved_mask_list.append(ori_mask)
            else:
                box_flow_map, mask = self.matcher(box_results, shape_list=shape_list, viz_flag=self.viz_bbx_flag) # [N_b, H, W, 2] [N_b, C, H, W]
            box_flow_map_list.append(box_flow_map)
            reserved_mask_list.append(mask)
            if self.diagnostic_roi_box_stats and dataset is not None:
                diag_aux_list.append(self._diagnose_single_roi_predictions(
                    box_results, data_dict, b, lidar_pose_batch[b], device,
                    single_score_stats))

            if self.viz_bbx_flag:
                single_box_results = box_results
        
        final_flow_map = torch.concat(box_flow_map_list, dim=0) # [N, H, W, 2] 一个batch中的合在一起
        final_reserved_mask = torch.concat(reserved_mask_list, dim=0)# [N, C, H, W] 一个batch中的合在一起
        comm_volum = sum(comm_volum) / B
        if self.diagnostic_roi_box_stats:
            self._last_roi_diagnostics = self._merge_box_diag_aux(
                diag_aux_list, device)

        if self.viz_bbx_flag:
            ori_reserved_mask = torch.concat(ori_reserved_mask_list, dim=0) 
            return final_flow_map, final_reserved_mask, ori_reserved_mask, single_box_results, matched_idx_list, compensated_results_list, comm_volum

        return final_flow_map, final_reserved_mask, comm_volum

    def forward(self, data_dict, dataset=None):
        voxel_features = data_dict['processed_lidar']['voxel_features']         #(M, 32, 4)
        voxel_coords = data_dict['processed_lidar']['voxel_coords']             #(M, 4)
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']     #(M, )
        record_len = data_dict['record_len']                                    #(B, )
        record_frames = data_dict['past_k_time_interval']                       #(sum(n_cav) ) batch中所有cav中每一帧到cur的时间间隔
        pairwise_t_matrix = data_dict['pairwise_t_matrix']                      #(B, L, k, 4, 4) 每一帧到cur ego
        
        # debug = 0
        B, _, k, _, _ = pairwise_t_matrix.shape
        # for i in range(B):
        #     debug += record_len[i]*k

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len} # (B) 每个场景下的车
        # n, 4 -> n, c  ('pillar_features')
        batch_dict = self.pillar_vfe(batch_dict)
        # (n, c) -> (batch_cav_size, C, H, W) put pillars into spatial feature map ('spatial_features')
        # import ipdb; ipdb.set_trace()
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict) # 'spatial_features_2d': (batch_cav_size, 128*3, H/2, W/2)
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d'] # 特征提取后的结果
        
        # print("spatial_features shape is :", batch_dict['spatial_features'].shape)
        # print("spatial_features_2d shape is :", spatial_features_2d.shape)

        shape_list = torch.tensor(batch_dict['spatial_features'].shape[-3:]).to(pairwise_t_matrix.device) # [64, 200, 704]

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


        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d).detach() # (B, 2, H', W')
        rm_single = self.reg_head(spatial_features_2d).detach() # (B, 14, H', W')

        # print("spatial_features_2d shape is :", spatial_features_2d.shape)
        # print("psm_single shape is :", psm_single.shape)
        # print("rm_single shape is :", rm_single.shape)
        # print("self.use_dir is :", self.use_dir)
        # print("self.design_mode is :", self.design_mode)
        
        if self.use_dir:
            dm_single = self.dir_head(spatial_features_2d)

        single_detection_bbx = None

        # if self.design_mode != 4 and not self.only_tune_header_flag:
        if self.design_mode != 4: 
            # generate box flow
            # [B, 256, 50, 176]
            single_output = {}
            single_output.update({'psm_single_list': self.regroup(psm_single, record_len, k), 
            'rm_single_list': self.regroup(rm_single, record_len, k)})
            if self.use_dir:
                single_output.update({
                    'dm_single_list': self.regroup(dm_single, record_len, k)
                })
            if self.viz_bbx_flag: # 可视化bbx
                box_flow_map, reserved_mask, ori_reserved_mask, single_detection_bbx, matched_idx_list, compensated_results_list, comm_volum = self.generate_box_flow(data_dict, single_output, dataset, shape_list, psm_single.device)
            else:
                box_flow_map, reserved_mask, comm_volum = self.generate_box_flow(data_dict, single_output, dataset, shape_list, psm_single.device) # 这是根据方向还有距离匹配past0和past1 再计算平均速度乘上时间得到的预估flow


        # print("box_flow_map shape is :", box_flow_map.shape)
        # print("reserved_mask shape is :", reserved_mask.shape)
        # print("batch_dict['spatial_features'] shape is :", batch_dict['spatial_features'].shape)
        


        if 'flow_gt' in data_dict['label_dict']:
            flow_gt = data_dict['label_dict']['flow_gt']
            mask_gt = data_dict['label_dict']['warp_mask']
        else:
            flow_gt = None
    
        # debug 在不使用运动预测的时候 默认先用GT流，GT流的生成为直接用 past0 与 cur下的object做
        # box_flow_map = flow_gt
        # reserved_mask = mask_gt
        
        # if self.only_tune_header_flag:
        #     box_flow_map = flow_gt
        #     reserved_mask = mask_gt

        # rain attention:
        if self.multi_scale:
            if self.design_mode == 0 or self.design_mode==5:
                if self.viz_bbx_flag: # 会额外返回 (BxN, C, H, W) 这是补偿之后的特征
                    fused_feature, communication_rates, result_dict, single_updated_feature = self.rain_fusion(batch_dict['spatial_features'],
                    psm_single,
                    record_len,
                    pairwise_t_matrix, 
                    record_frames,
                    self.fused_backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head],
                    box_flow=box_flow_map, reserved_mask=reserved_mask,
                    flow_gt=flow_gt, viz_bbx_flag=self.viz_bbx_flag)
                else:
                    fused_feature, communication_rates, result_dict = self.rain_fusion(batch_dict['spatial_features'], # (sum(n_cav), C, H, W) 所有帧的
                        psm_single, # (B, 2, H', W')
                        record_len, # (B, )
                        pairwise_t_matrix, # (B, L, k, 4, 4)
                        record_frames,# (B, )
                        self.fused_backbone, # ResNetBEVBackbone(args['base_bev_backbone'], 64)
                        [self.shrink_conv, self.cls_head, self.reg_head],
                        box_flow=box_flow_map, reserved_mask=reserved_mask,
                        flow_gt=flow_gt, viz_bbx_flag=self.viz_bbx_flag, noise_pairwise_t_matrix=noise_pairwise_t_matrix)
            elif self.design_mode == 4:
                fused_feature, communication_rates, result_dict = self.rain_fusion(batch_dict['spatial_features'],
                    psm_single,
                    record_len,
                    pairwise_t_matrix, 
                    record_frames,
                    self.backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head])
            else: 
                fused_feature, communication_rates, result_dict, flow_recon_loss = self.rain_fusion(batch_dict['spatial_features'],
                    psm_single,
                    record_len,
                    pairwise_t_matrix, 
                    record_frames,
                    self.backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head],
                    box_flow=box_flow_map, reserved_mask=reserved_mask,
                    flow_gt=flow_gt)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.fused_shrink_conv(fused_feature) # (B, 256, H/2, W/2)
                if self.single_supervise:
                    single_feature = self.shrink_conv(single_feature)

        else:
            fused_feature, communication_rates, result_dict = self.rain_fusion(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix,
                                            record_frames)
            if self.compensation:
                if self.single_supervise:
                    fused_feature, single_feature, communication_rates, all_recon_loss, result_dict = self.rain_fusion(spatial_features_2d,
                                                psm_single,
                                                record_len,
                                                pairwise_t_matrix, 
                                                record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])
                else:
                    fused_feature,fused_feature_curr,fused_feature_latency, communication_rates, all_recon_loss, all_latency_recon_loss, result_dict = self.rain_fusion(spatial_features_2d,
                                                psm_single,
                                                record_len,
                                                pairwise_t_matrix, 
                                                record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])            

        
        # print('fused_feature: ', fused_feature.shape)
        # exit9

        if self.only_tune_header_flag:
            psm = self.fused_cls_head(fused_feature)
            rm = self.fused_reg_head(fused_feature)
        else: 
            # psm = self.cls_head(fused_feature) # 默认是开启下方这个，这里先注释 2024年04月20日
            # rm = self.reg_head(fused_feature)
            psm = self.fused_cls_head(fused_feature)
            rm = self.fused_reg_head(fused_feature)
        
        output_dict = {'psm': psm,
                       'rm': rm}

        if self.use_dir:
            # dm = self.dir_head(fused_feature)
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
                       'comm_rate': comm_volum
                       })
        if self.diagnostic_roi_box_stats:
            output_dict.update(getattr(self, '_last_roi_diagnostics', {}))

        if self.viz_bbx_flag:
            output_dict.update({
                'single_detection_bbx': single_detection_bbx, # Dict 包含了三帧的detect结果
                'matched_idx_list': matched_idx_list,  # List [(N_obj_cav1, 2/3), ...]
                'compensated_results_list': compensated_results_list # List [(N_obj_cav1, 4, 2), ...]
            })
            _, C, H, W = batch_dict['spatial_features'].shape
            output_dict.update({
                'single_updated_feature': single_updated_feature, # 补偿后的特征 (BxN, C, H, W)
                'single_original_feature': batch_dict['spatial_features'].reshape(-1, self.k, C, H, W)[:, 0, :, :, :],  # 没有特征提取前的特征（BxN, 0, C, H, W） past0 其中
                'single_flow_map': box_flow_map,  # 生成的流  [N, H, W, 2] 
                'single_reserved_mask': reserved_mask,  # 流掩码  [N, C, H, W]
                'single_original_reserved_mask': ori_reserved_mask # 和上面的流掩码形状一样，但是是将原始past0中的object的区域都置1
            })

        if self.design_mode == 1:
            output_dict.update({'flow_recon_loss': flow_recon_loss})
        
        output_dict.update(result_dict) 
        
        return output_dict
