"""CoST voxel post-processing port for OpenCOOD collated batches."""

import torch

from opencood.utils import box_utils

try:
    from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import nms_gpu as _cost_nms_gpu
except ImportError:
    _cost_nms_gpu = None


class CostPostProcessor:
    """CoST decode -> project -> rotated-NMS pipeline."""

    def __init__(self, params):
        self.params = params

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        batch = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(batch, -1, 7)
        anchors = anchors.to(device=deltas.device, dtype=torch.float32).view(-1, 7)
        diagonal = torch.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)
        diagonal = diagonal.repeat(batch, 2, 1).transpose(1, 2)
        anchors = anchors.repeat(batch, 1, 1)
        boxes = torch.zeros_like(deltas)
        boxes[..., [0, 1]] = deltas[..., [0, 1]] * diagonal + anchors[..., [0, 1]]
        boxes[..., [2]] = deltas[..., [2]] * anchors[..., [3]] + anchors[..., [2]]
        boxes[..., [3, 4, 5]] = torch.exp(deltas[..., [3, 4, 5]]) * anchors[..., [3, 4, 5]]
        boxes[..., 6] = deltas[..., 6] + anchors[..., 6]
        return boxes

    def post_process(self, data_dict, output_dict):
        ego = data_dict["ego"]
        output = output_dict["ego"]
        anchors = ego["anchor_box"]
        if anchors.dim() == 5:
            anchors = anchors[0]
        transform = ego.get("transformation_matrix", torch.eye(4, device=anchors.device))
        if transform.dim() == 3:
            transform = transform[0]
        probability = torch.sigmoid(output["psm"].permute(0, 2, 3, 1)).reshape(1, -1)
        boxes = self.delta_to_boxes3d(output["rm"], anchors)[0]
        mask = probability[0] > self.params["target_args"]["score_threshold"]
        boxes, scores = boxes[mask], probability[0][mask]
        if boxes.numel() == 0:
            return None, None
        identity = torch.eye(4, device=transform.device, dtype=transform.dtype)
        use_cuda_nms = _cost_nms_gpu is not None and torch.allclose(transform, identity)
        if use_cuda_nms:
            # CoST stores dimensions as h,w,l; pcdet CUDA NMS expects l,w,h.
            nms_boxes = boxes[:, [0, 1, 2, 5, 4, 3, 6]].contiguous()
            keep = _cost_nms_gpu(
                nms_boxes,
                scores.contiguous(),
                thresh=self.params["nms_thresh"],
                pre_maxsize=self.params.get("nms_pre_maxsize", None),
            )[0]
            keep = keep.to(device=boxes.device, dtype=torch.long)
            corners = box_utils.boxes_to_corners_3d(boxes, order=self.params["order"])
        else:
            corners = box_utils.boxes_to_corners_3d(boxes, order=self.params["order"])
            corners = box_utils.project_box3d(corners, transform.to(torch.float32))
            keep = box_utils.nms_rotated(corners, scores, self.params["nms_thresh"])
            keep = torch.as_tensor(keep, device=boxes.device, dtype=torch.long)
        corners, scores = corners[keep], scores[keep]
        within = box_utils.get_mask_for_boxes_within_range_torch(
            corners, self.params["anchor_args"]["cav_lidar_range"]
        )
        return corners[within], scores[within]
