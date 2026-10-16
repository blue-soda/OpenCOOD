"""CPU-only paired post-NMS diagnosis; reproduce evaluator matching exactly."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from opencood.utils import common_utils, eval_utils


def bev_error(pred, gt):
    """Corner-derived BEV geometry; long-axis heading modulo pi, not direction."""
    def geometry(box):
        xy = box[:4, :2].astype(np.float64)
        edges = np.asarray([xy[1] - xy[0], xy[2] - xy[1]])
        lengths = np.linalg.norm(edges, axis=1)
        major = int(np.argmax(lengths))
        return xy.mean(axis=0), np.sort(lengths), np.arctan2(edges[major, 1], edges[major, 0])
    pc, ps, py = geometry(pred)
    gc, gs, gy = geometry(gt)
    yaw = abs((py - gy + np.pi / 2) % np.pi - np.pi / 2)
    return dict(center_error_m=float(np.linalg.norm(pc - gc)),
                length_error_m=float(abs(ps[1] - gs[1])),
                width_error_m=float(abs(ps[0] - gs[0])),
                axis_error_deg=float(np.degrees(yaw)))


def pr_summary(stats):
    ap, recall, precision = eval_utils.calculate_ap({.7: stats}, .7)
    recall, precision = np.asarray(recall), np.asarray(precision)
    intervals = {}
    for low, high in ((0., .3), (.3, .5), (.5, 1.)):
        width = np.maximum(0., np.minimum(recall[1:], high) - np.maximum(recall[:-1], low))
        intervals['{}:{}'.format(low, high)] = float(np.sum(width * precision[1:]))
    if not np.isclose(sum(intervals.values()), ap, atol=1e-12, rtol=0):
        raise ValueError('PR interval integration differs from AP')
    at_recall = {}
    for target in (.1, .2, .3, .4, .5, .55, .6, .62, .64):
        at_recall[str(target)] = float(precision[recall >= target].max())
    return dict(ap70=ap, precision_envelope_at_recall=at_recall, ap70_by_recall_interval=intervals)


def match(iou, scores, threshold):
    """Score-ordered greedy matching to remaining GT, as in eval_utils."""
    order = np.argsort(-scores)
    remaining = list(range(iou.shape[1]))
    tp, assigned = [], []
    for pred in order:
        if not remaining or np.max(iou[pred, remaining]) < threshold:
            tp.append(0)
            assigned.append(-1)
        else:
            gt = remaining.pop(int(np.argmax(iou[pred, remaining])))
            tp.append(1)
            assigned.append(gt)
    return order, np.asarray(tp, dtype=int), assigned


def read_index(folder):
    report = json.loads((folder / 'evaluation.json').read_text())
    path = folder / 'predictions/index.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != report['predictions']['index_sha256']:
        raise ValueError('Index hash mismatch')
    rows = json.loads(path.read_text())['frames']
    if len(rows) != report['samples'] or len({x['frame_id'] for x in rows}) != len(rows):
        raise ValueError('Incomplete/duplicate frame index')
    return report, rows, digest


def read_frame(folder, row):
    path = folder / 'predictions' / row['file']
    if path.resolve().parent != (folder / 'predictions').resolve():
        raise ValueError('Invalid prediction path')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Frame hash mismatch')
    with np.load(str(path), allow_pickle=False) as data:
        if str(data['frame_id'].item()) != row['frame_id']:
            raise ValueError('Frame ID mismatch')
        pred, score, gt = [data[k].copy() for k in ('pred_boxes', 'scores', 'gt_boxes')]
    if pred.shape != (row['predicted_boxes'], 8, 3) or gt.shape != (row['gt_boxes'], 8, 3):
        raise ValueError('Frame shape mismatch')
    if score.shape != (len(pred),) or not all(np.isfinite(x).all() for x in (pred, score, gt)):
        raise ValueError('Invalid frame values')
    return pred, score, gt


def diagnose(ego_dir, fusion_dir, expected_frames=1738):
    folders = dict(ego=Path(ego_dir), fusion=Path(fusion_dir))
    bundles = {name: read_index(folder) for name, folder in folders.items()}
    if any(len(x[1]) != expected_frames for x in bundles.values()):
        raise ValueError('Unexpected validation split length')
    stats = {n: {t: dict(tp=[], fp=[], score=[], gt=0) for t in (.5, .7)} for n in folders}
    counts = {n: dict(predicted=0, gt=0, tp70=0, fp70=0,
                     fp_best_iou_below_03=0, fp_best_iou_03_to_07=0,
                     fp_best_iou_at_least_07=0, gt_any_iou50=0, gt_any_iou70=0)
              for n in folders}
    transitions = dict(retained=0, lost=0, gained=0, missed_by_both=0)
    details = []
    lost_examples = []
    for position in range(expected_frames):
        rows = {n: bundles[n][1][position] for n in folders}
        if any(r['index'] != position for r in rows.values()) or rows['ego']['frame_id'] != rows['fusion']['frame_id']:
            raise ValueError('Frame order mismatch')
        frames = {n: read_frame(folders[n], rows[n]) for n in folders}
        if not np.array_equal(frames['ego'][2], frames['fusion'][2]):
            raise ValueError('GT arrays differ')
        matched, best_gt, best_pred = {}, {}, {}
        for name, (pred, scores, gt) in frames.items():
            polys = list(common_utils.convert_format(gt))
            matrix = np.asarray([common_utils.compute_iou(p, polys)
                                 for p in common_utils.convert_format(pred)]).reshape(len(pred), len(gt))
            c = counts[name]
            c['predicted'] += len(pred)
            c['gt'] += len(gt)
            best_gt[name] = matrix.max(axis=0) if len(pred) else np.zeros(len(gt))
            best_pred[name] = matrix.argmax(axis=0) if len(pred) else np.full(len(gt), -1)
            c['gt_any_iou50'] += int((best_gt[name] >= .5).sum())
            c['gt_any_iou70'] += int((best_gt[name] >= .7).sum())
            for threshold in (.5, .7):
                order, tp, assigned = match(matrix, scores, threshold)
                s = stats[name][threshold]
                s['tp'].extend(tp.tolist())
                s['fp'].extend((1 - tp).tolist())
                s['score'].extend(scores[order].tolist())
                s['gt'] += len(gt)
                if threshold == .7:
                    matched[name] = {x for x in assigned if x >= 0}
                    c['tp70'] += int(tp.sum())
                    c['fp70'] += int((1 - tp).sum())
                    best = matrix.max(axis=1) if len(gt) else np.zeros(len(pred))
                    fp_iou = best[order][tp == 0]
                    c['fp_best_iou_below_03'] += int((fp_iou < .3).sum())
                    c['fp_best_iou_03_to_07'] += int(((fp_iou >= .3) & (fp_iou < .7)).sum())
                    c['fp_best_iou_at_least_07'] += int((fp_iou >= .7).sum())
        a, b = matched['ego'], matched['fusion']
        row = dict(index=position, frame_id=rows['ego']['frame_id'],
                   retained=len(a & b), lost=len(a - b), gained=len(b - a),
                   missed_by_both=len(frames['ego'][2]) - len(a | b))
        for key in transitions:
            transitions[key] += row[key]
        details.append(row)
        for gt_id in sorted(a - b):
            item = dict(frame_id=row['frame_id'], gt_index=gt_id,
                        ego_best_iou=float(best_gt['ego'][gt_id]),
                        fusion_best_iou=float(best_gt['fusion'][gt_id]))
            if .5 <= best_gt['fusion'][gt_id] < .7:
                item['geometry'] = {n: bev_error(frames[n][0][best_pred[n][gt_id]], frames[n][2][gt_id])
                                    for n in folders}
                item['gt_range_m'] = float(np.linalg.norm(frames['ego'][2][gt_id, :4, :2].mean(axis=0)))
            lost_examples.append(item)
        if (position + 1) % 250 == 0:
            print(json.dumps(dict(diagnosed=position + 1)), flush=True)
    models = {}
    for name in folders:
        ap = {str(t): eval_utils.calculate_ap(stats[name], t)[0] for t in (.5, .7)}
        if any(ap[str(t)] != bundles[name][0]['ap'][str(t)] for t in (.5, .7)):
            raise ValueError('Diagnostic matching did not exactly reproduce archived AP')
        if counts[name]['predicted'] != bundles[name][0]['predicted_boxes'] or counts[name]['gt'] != bundles[name][0]['gt_boxes']:
            raise ValueError('Box count mismatch')
        c = counts[name]
        operating_points = {}
        s = stats[name][.7]
        for cutoff in (.2, .5, .7, .9):
            keep = np.asarray(s['score']) >= cutoff
            tp = int(np.asarray(s['tp'])[keep].sum())
            fp = int(np.asarray(s['fp'])[keep].sum())
            operating_points[str(cutoff)] = dict(tp=tp, fp=fp, recall=tp/c['gt'], precision=tp/max(tp+fp, 1))
        models[name] = dict(directory=str(folders[name]), index_sha256=bundles[name][2],
                            ap=ap, counts=c, recall70=c['tp70']/c['gt'],
                            precision70=c['tp70']/max(c['predicted'], 1),
                            score_thresholds=operating_points, pr70=pr_summary(s))
    lost_examples.sort(key=lambda x: x['fusion_best_iou'])
    localized_lost = [x for x in lost_examples if 'geometry' in x]
    geometry = dict(count=len(localized_lost),
        scope='GT matched by ego at IoU .7, not by fusion; fusion best IoU in [.5,.7). '
              'Compare each model best-IoU prediction, not necessarily score-greedy assignment. '
              'Outcome-selected cohort; does not estimate population-wide or causal module effects.')
    if localized_lost:
        for key in ('center_error_m', 'length_error_m', 'width_error_m', 'axis_error_deg'):
            a = np.asarray([x['geometry']['ego'][key] for x in localized_lost])
            b = np.asarray([x['geometry']['fusion'][key] for x in localized_lost])
            geometry[key] = dict(ego_median=float(np.median(a)), fusion_median=float(np.median(b)),
                                 median_paired_change=float(np.median(b - a)),
                                 fraction_worse=float(np.mean(b > a)))
    return dict(passed=True, frames=expected_frames, models=models, gt_transitions70=transitions,
                per_frame=details, lost_examples=lost_examples, localized_lost_geometry=geometry,
                scope='Post-NMS saved predictions only; exact repository score-greedy BEV matching. '
                      'Any-IoU counts ignore score/one-to-one assignment. FP categories are descriptive, '
                      'not causal attribution to quantization, alignment, score calibration or NMS.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ego-dir', required=True)
    parser.add_argument('--fusion-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-frames', type=int, default=1738)
    args = parser.parse_args()
    result = diagnose(args.ego_dir, args.fusion_dir, args.expected_frames)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=True, transitions=result['gt_transitions70'], models=result['models'])), flush=True)
