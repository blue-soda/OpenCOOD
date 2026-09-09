# -*- coding: utf-8 -*-
"""V2V4Real reader for CoBEVFlow-style irregular asynchronous training."""

import copy
import numpy as np

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
        super().__init__(params=params, visualize=visualize, train=train)

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
