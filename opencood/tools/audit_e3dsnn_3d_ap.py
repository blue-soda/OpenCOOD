"""Check supplemental 3D AP geometry, matching, and independent CUDA IoU."""
import argparse
import json
import numpy as np
import torch
from opencood.utils import box_utils, eval_utils, eval_3d_utils


def new_stat():
    return {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in (.3, .5, .7)}


def run():
    base = np.array([[0., 0., 0., 4., 2., 2., 0.]], dtype=np.float32)
    variants = np.repeat(base, 5, axis=0)
    variants[1, 2] = 3.  # identical footprint, disjoint heights
    variants[2, 2] = 1.  # half-height intersection: 1/3 IoU
    variants[3, 0] = 10.  # horizontal separation
    variants[4, 6] = np.pi / 2  # crossed 4x2 rectangles: 1/3 IoU
    gt = box_utils.boxes_to_corners_3d(base, 'lwh')
    boxes = box_utils.boxes_to_corners_3d(variants, 'lwh')
    np.testing.assert_allclose(eval_3d_utils.pairwise_iou3d(boxes, gt)[:, 0],
                               [1., 0., 1. / 3, 0., 1. / 3], atol=1e-6)
    empty = np.empty((0, 8, 3), dtype=np.float32)
    assert eval_3d_utils.pairwise_iou3d(empty, gt).shape == (0, 1)
    assert eval_3d_utils.pairwise_iou3d(gt, empty).shape == (1, 0)
    malformed = gt.copy()
    malformed[0, 0, 2] += .1
    try:
        eval_3d_utils.pairwise_iou3d(malformed, gt)
    except ValueError:
        pass
    else:
        raise AssertionError('Non-upright boxes should fail explicitly')
    scores = torch.tensor([.8, .9, .7])
    # The high-confidence elevated box is FP; one correct box TP; duplicate FP.
    predictions = torch.from_numpy(boxes[[0, 1, 0]])
    stat = new_stat()
    eval_3d_utils.calculate_tp_fp_3d(predictions, scores, torch.from_numpy(gt), stat)
    for threshold in stat:
        assert stat[threshold]['tp'] == [0, 1, 0]
        assert stat[threshold]['fp'] == [1, 0, 1]
        assert eval_utils.calculate_ap(stat, threshold)[0] == .5
    bev = new_stat()
    for threshold in bev:
        eval_utils.caluclate_tp_fp(predictions, scores, torch.from_numpy(gt), bev, threshold)
        assert eval_utils.calculate_ap(bev, threshold)[0] == 1.
    # No predictions still contributes all ground truth; no GT makes every box FP.
    no_predictions = new_stat()
    eval_3d_utils.calculate_tp_fp_3d(None, None, gt, no_predictions)
    no_gt = new_stat()
    eval_3d_utils.calculate_tp_fp_3d(predictions, scores, empty, no_gt)
    assert no_predictions[.5]['gt'] == 1 and no_predictions[.5]['tp'] == []
    assert no_gt[.5]['fp'] == [1, 1, 1] and no_gt[.5]['gt'] == 0
    # Validate corner convention, rotation and vertical extent against an
    # independent existing CUDA implementation using center/size/yaw boxes.
    from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
    rng = np.random.RandomState(20260929)
    samples = np.zeros((96, 7), dtype=np.float32)
    samples[:, :3] = rng.uniform(-3., 3., (96, 3))
    samples[:, 3:6] = rng.uniform(.5, 5., (96, 3))
    samples[:, 6] = rng.uniform(-np.pi, np.pi, 96)
    corners = box_utils.boxes_to_corners_3d(samples, 'lwh')
    actual = eval_3d_utils.pairwise_iou3d(corners[:48], corners[48:])
    expected = boxes_iou3d_gpu(torch.from_numpy(samples[:48]).cuda(),
                               torch.from_numpy(samples[48:]).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
    return {'passed': True, 'synthetic_geometry': 'identity, height separation, partial height, xy separation, yaw',
            'matching': 'score order, duplicate rejection, empty predictions/GT, AP integration',
            'legacy_bev_ap': 1., 'same_boxes_3d_ap': .5,
            'cuda_reference_pairs': int(actual.size),
            'cuda_max_abs_error': float(np.max(np.abs(actual - expected)))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run()
    with open(args.output, 'w') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
