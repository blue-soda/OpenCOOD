# -*- coding: utf-8 -*-
"""V2V4Real reader for CoBEVFlow-style irregular asynchronous training."""

import copy
import numpy as np
import torch

from opencood.data_utils.datasets.intermediate_fusion_dataset_opv2v_irregular import (
    IntermediateFusionDatasetIrregular,
)
from opencood.utils import box_utils, common_utils


class CoBEVFlowV2V4RealIrregularDataset(IntermediateFusionDatasetIrregular):
    """CoBEVFlow OPV2V-irregular protocol with V2V4Real label semantics.

    V2V4Real yaml metadata stores ``lidar_pose`` as a 4x4 transform matrix, and
    its object labels are interpreted in the CAV-local lidar frame. The parent
    class keeps the CoBEVFlow Bernoulli frame-delay protocol; this subclass only
    makes the V2V4Real coordinate convention explicit.
    """

    def __init__(self, params, visualize, train=True):
        params = copy.deepcopy(params)
        params["object_label_coord"] = "local"
        self.iou_dedup_threshold = params.get("v2v4real_iou_dedup_threshold", 0.05)
        self.is_generate_motion_gt = params.get("is_generate_motion_gt", False)
        self.motion_gt_only = self.is_generate_motion_gt and \
            params.get("motion_gt_only", False)
        self.motion_match_max_dist = params.get("motion_match_max_dist", 8.0)
        super().__init__(params=params, visualize=visualize, train=train)

    def __getitem__(self, idx):
        if self.motion_gt_only:
            base_data_dict = self.retrieve_base_data(idx)
            return self._build_motion_gt_only_sample(base_data_dict)
        return super().__getitem__(idx)

    def _build_motion_gt_only_sample(self, base_data_dict):
        past_k_object_bbx_stack = []
        past_k_cav_object_num = []
        cur_cav_object_bbx_debug = []
        past_k_time_diffs_stack = []
        past_k_sample_interval_stack = []

        for _, selected_cav_base in base_data_dict.items():
            if selected_cav_base['ego']:
                continue

            past0_pose = selected_cav_base['past_k'][0]['params']['lidar_pose']
            past_k_object_bbx = []
            for time_idx in range(self.k):
                past_k_object_bbx.append(
                    self._generate_boxes_in_pose(
                        selected_cav_base['past_k'][time_idx], past0_pose))
            cur_object_bbx = self._generate_boxes_in_pose(
                selected_cav_base['curr'], past0_pose)

            past_k_common_bbx, cur_common_bbx = \
                self._match_temporal_boxes_by_nearest(
                    past_k_object_bbx, cur_object_bbx)
            if past_k_common_bbx is None or past_k_common_bbx.shape[0] == 0:
                continue

            past_k_object_bbx_stack.append(past_k_common_bbx)
            past_k_cav_object_num.append(past_k_common_bbx.shape[0])
            cur_cav_object_bbx_debug.append(cur_common_bbx)
            past_k_time_diffs_stack += [
                selected_cav_base['past_k'][time_idx]['time_diff']
                for time_idx in range(self.k)]
            past_k_sample_interval_stack += [
                selected_cav_base['past_k'][time_idx]['sample_interval']
                for time_idx in range(self.k)]

        if len(past_k_object_bbx_stack) == 0:
            return None

        past_k_sample_interval_array = np.array(past_k_sample_interval_stack)
        past_k_time_diffs_array = np.array(past_k_time_diffs_stack)
        return {'ego': {
            'label_dict': {},
            'past_k_object_bbx': np.vstack(past_k_object_bbx_stack),
            'past_k_cav_object_num': past_k_cav_object_num,
            'cur_cav_object_bbx_debug': np.vstack(cur_cav_object_bbx_debug),
            'past_k_time_diffs': past_k_time_diffs_array,
            'past_k_sample_interval': past_k_sample_interval_array,
            'avg_sample_interval': float(np.mean(past_k_sample_interval_array)),
            'avg_time_delay': float(np.mean(past_k_time_diffs_array)),
            'avg_var': float(np.var(past_k_time_diffs_array)),
        }}

    def _generate_boxes_in_pose(self, frame_content, reference_pose):
        object_bbx_center, object_bbx_mask, _ = \
            self.generate_object_center([frame_content], reference_pose)
        return object_bbx_center[object_bbx_mask == 1]

    def _match_temporal_boxes_by_nearest(self, past_k_boxes, cur_boxes):
        if not past_k_boxes or past_k_boxes[0].shape[0] == 0 or \
                cur_boxes.shape[0] == 0:
            return None, None

        matched_past = []
        matched_cur = []
        base_boxes = past_k_boxes[0]
        for base_box in base_boxes:
            track_boxes = []
            ok = True
            for boxes in past_k_boxes:
                if boxes.shape[0] == 0:
                    ok = False
                    break
                dist = np.linalg.norm(boxes[:, :2] - base_box[:2], axis=1)
                nearest_idx = int(np.argmin(dist))
                if dist[nearest_idx] > self.motion_match_max_dist:
                    ok = False
                    break
                track_boxes.append(boxes[nearest_idx])
            if not ok:
                continue

            cur_dist = np.linalg.norm(cur_boxes[:, :2] - base_box[:2], axis=1)
            cur_idx = int(np.argmin(cur_dist))
            if cur_dist[cur_idx] > self.motion_match_max_dist:
                continue
            matched_past.append(np.stack(track_boxes, axis=0))
            matched_cur.append(cur_boxes[cur_idx])

        if len(matched_past) == 0:
            return None, None
        return np.stack(matched_past, axis=0), np.stack(matched_cur, axis=0)

    def collate_batch_train(self, batch):
        if not self.motion_gt_only:
            return super().collate_batch_train(batch)

        batch = [sample for sample in batch if sample is not None]
        if len(batch) == 0:
            return None

        past_k_object_bbx_list = []
        past_k_object_cav_num_list = []
        cur_object_bbx_debug_list = []
        past_k_time_diff = []
        past_k_sample_interval = []
        avg_sample_interval = []
        avg_time_delay = []
        avg_time_var = []
        for sample in batch:
            ego_dict = sample['ego']
            past_k_object_bbx_list.append(ego_dict['past_k_object_bbx'])
            past_k_object_cav_num_list += ego_dict['past_k_cav_object_num']
            cur_object_bbx_debug_list.append(ego_dict['cur_cav_object_bbx_debug'])
            past_k_time_diff.append(ego_dict['past_k_time_diffs'])
            past_k_sample_interval.append(ego_dict['past_k_sample_interval'])
            avg_sample_interval.append(ego_dict['avg_sample_interval'])
            avg_time_delay.append(ego_dict['avg_time_delay'])
            avg_time_var.append(ego_dict['avg_var'])

        return {'ego': {
            'label_dict': {},
            'past_k_object_bbx': torch.from_numpy(
                np.vstack(past_k_object_bbx_list)),
            'past_k_object_cav_num': torch.from_numpy(
                np.array(past_k_object_cav_num_list)),
            'cur_object_bbx_debug': torch.from_numpy(
                np.vstack(cur_object_bbx_debug_list)),
            'past_k_time_interval': torch.from_numpy(
                np.hstack(past_k_time_diff)),
            'past_k_sample_interval': torch.from_numpy(
                np.hstack(past_k_sample_interval)),
            'avg_sample_interval': float(np.mean(avg_sample_interval)),
            'avg_time_delay': float(np.mean(avg_time_delay)),
            'avg_time_var': float(np.mean(avg_time_var)),
        }}

    def collate_batch_test(self, batch):
        if self.motion_gt_only:
            return self.collate_batch_train(batch)
        return super().collate_batch_test(batch)

    def deduplicate_object_stack(self, object_stack, object_id_stack):
        """Deduplicate V2V4Real local labels by projected BEV overlap.

        V2V4Real per-CAV yaml files can use local, per-agent object ids. After
        all boxes are projected to ego coordinates, id-based deduplication can
        either keep duplicates with different ids or drop distinct boxes whose
        local ids happen to collide. BEV IoU is the safer signal for Stage-1
        detector supervision.
        """
        if object_stack.shape[0] <= 1:
            return object_stack, list(object_id_stack)

        corners = box_utils.boxes_to_corners_3d(
            object_stack, self.params["postprocess"]["order"])
        polygons = list(common_utils.convert_format(corners))
        keep_indices = []

        for idx, polygon in enumerate(polygons):
            if not keep_indices:
                keep_indices.append(idx)
                continue
            kept_polygons = [polygons[keep_idx] for keep_idx in keep_indices]
            ious = common_utils.compute_iou(polygon, kept_polygons)
            if len(ious) == 0 or float(np.max(ious)) <= self.iou_dedup_threshold:
                keep_indices.append(idx)

        unique_object_ids = [
            "v2v4real_iou_%03d_%s" % (rank, object_id_stack[idx])
            for rank, idx in enumerate(keep_indices)
        ]
        return object_stack[keep_indices], unique_object_ids
