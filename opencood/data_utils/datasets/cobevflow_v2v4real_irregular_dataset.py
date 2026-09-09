# -*- coding: utf-8 -*-
"""V2V4Real reader for CoBEVFlow-style irregular asynchronous training."""

import copy

from opencood.data_utils.datasets.intermediate_fusion_dataset_opv2v_irregular import (
    IntermediateFusionDatasetIrregular,
)


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
        super().__init__(params=params, visualize=visualize, train=train)
