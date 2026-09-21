"""TraF-Align anchor decode and rotated-NMS post-processing port."""

import torch

from opencood.utils import box_utils

try:
    from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import nms_gpu as _traf_nms_gpu
except ImportError:
    # The server's OpenCOOD environment does not ship nvcc-built pcdet
    # extensions.  Keep the original CUDA path when available and fall back
    # to the repository's geometry implementation otherwise.
    _traf_nms_gpu = None


class TrafResidualCoder:
    @staticmethod
    def decode_torch(encodings, anchors):
        xa, ya, za, dxa, dya, dza, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, dxt, dyt, dzt, rt = torch.split(encodings, 1, dim=-1)
        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg, yg = xt * diagonal + xa, yt * diagonal + ya
        zg = zt * dza + za
        dxg, dyg, dzg = torch.exp(dxt) * dxa, torch.exp(dyt) * dya, torch.exp(dzt) * dza
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rt + ra], dim=-1)


class TrafPostProcessor:
    """Port of TraF's AnchorProcessor for the migrated adapter outputs."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.post_cfg = cfg["post_processing"]
        self.anchor_cfg = cfg["model"]["head"]["anchor_generator_config"]

    def _anchors(self, height, width, device):
        point_range = self.cfg["voxelization"]["lidar_range"]
        anchors = []
        for config in self.anchor_cfg:
            x_stride = (point_range[3] - point_range[0]) / max(width - 1, 1)
            y_stride = (point_range[4] - point_range[1]) / max(height - 1, 1)
            xs = torch.arange(width, device=device) * x_stride + point_range[0]
            ys = torch.arange(height, device=device) * y_stride + point_range[1]
            yy, xx = torch.meshgrid(ys, xs)
            for size in config["anchor_sizes"]:
                for bottom in config["anchor_bottom_heights"]:
                    for rotation in config["anchor_rotations"]:
                        z = torch.full_like(xx, bottom + size[2] / 2.0)
                        anchors.append(torch.stack([
                            xx, yy, z,
                            torch.full_like(xx, size[0]),
                            torch.full_like(xx, size[1]),
                            torch.full_like(xx, size[2]),
                            torch.full_like(xx, rotation),
                        ], dim=-1))
        return torch.stack(anchors, dim=2)

    def post_process(self, data_dict, output_dict):
        output = output_dict["ego"]
        cls, reg = output["psm"], output["rm"]
        batch, _, height, width = cls.shape
        anchors = self._anchors(height, width, cls.device)
        anchors = anchors.view(1, height, width, -1, 7).repeat(batch, 1, 1, 1, 1)
        scores = torch.sigmoid(cls.permute(0, 2, 3, 1)).reshape(batch, -1)
        encodings = reg.permute(0, 2, 3, 1).reshape(batch, -1, 7)
        boxes = TrafResidualCoder.decode_torch(encodings, anchors.reshape(batch, -1, 7))
        results = []
        for index in range(batch):
            score, box = scores[index], boxes[index]
            mask = score > self.post_cfg["score_thresh"]
            score, box = score[mask], box[mask]
            if box.numel() == 0:
                results.append({"box3d_lidar": box, "scores": score})
                continue
            if _traf_nms_gpu is not None:
                keep = _traf_nms_gpu(
                    box.contiguous(),
                    score.contiguous(),
                    thresh=self.post_cfg["score_thresh"],
                    pre_maxsize=self.post_cfg["nms_config"]["nms_pre_maxsize"],
                    post_maxsize=self.post_cfg["nms_config"]["nms_post_maxsize"],
                )[0]
                keep = keep.to(device=box.device, dtype=torch.long)
            else:
                corners = box_utils.boxes_to_corners_3d(box, order="lhw")
                keep = box_utils.nms_rotated(corners, score, self.post_cfg["score_thresh"])
                keep = torch.as_tensor(keep, device=box.device, dtype=torch.long)
            results.append({
                "box3d_lidar": box[keep][:, [0, 1, 2, 5, 4, 3, 6]],
                "scores": score[keep],
            })
        return results
