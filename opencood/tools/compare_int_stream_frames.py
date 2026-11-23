"""Compare two complete scored DAIR streams by scene and five-scan window position."""
import argparse
import json
from pathlib import Path

import numpy as np


def point(records):
    if not records:
        return None
    gt = sum(x['ap70']['gt'] for x in records)
    if gt == 0:
        return None
    scores = np.concatenate([x['ap70']['score'] for x in records])
    tp = np.concatenate([x['ap70']['tp'] for x in records]).astype(bool)[np.argsort(-scores)]
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    ap = np.maximum.accumulate(precision[::-1])[::-1][tp].sum() / gt
    return dict(ap70=float(ap), frames=len(records), gt=gt, weak=sum(x['weak'] for x in records),
                weak_hits=sum(x['hits'] for x in records))


def compare(left, right, manifest):
    def key(r):
        return r['scene'], r['frame']
    phase = {}
    old_segment = None
    age = 0
    for row in manifest:
        if row['segment'] != old_segment or row.get('reset_before'):
            age = 0
        phase[key(row)] = age
        age = (age + 1) % 5
        old_segment = row['segment']
    expected = [key(row) for row in manifest if row['supervised']]
    assert list(map(key, left)) == list(map(key, right)) == expected
    assert all(a['weak'] == b['weak'] and a['ap70']['gt'] == b['ap70']['gt'] for a,b in zip(left,right))
    report = {}
    for label, ages in [('all', range(5)), ('anchor', (0,)), ('ordinary', range(1,5))]:
        report[label] = [point([r for r in rows if phase[key(r)] in ages]) for rows in (left,right)]
    scenes = list(dict.fromkeys(r['scene'] for r in left))
    def group(rows):
        result = []
        for scene in scenes:
            observed = [r['ap70'] for r in rows if r['scene'] == scene]
            result.append((np.concatenate([x['score'] for x in observed]),
                           np.concatenate([x['tp'] for x in observed]).astype(bool), sum(x['gt'] for x in observed)))
        return result
    def pooled(groups, draw):
        scores = np.concatenate([groups[i][0] for i in draw])
        tp = np.concatenate([groups[i][1] for i in draw])[np.argsort(-scores)]
        precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
        return np.maximum.accumulate(precision[::-1])[::-1][tp].sum() / sum(groups[i][2] for i in draw)
    a,b = group(left),group(right)
    draws = np.random.RandomState(303).randint(len(scenes), size=(2000, len(scenes)))
    report['delta_ap70_pp_ci95_left_minus_right'] = np.percentile(
        [100 * (pooled(a, draw) - pooled(b, draw)) for draw in draws], [2.5,97.5]).tolist()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('left', 'right', 'plan', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]
    result = compare(rows(args.left), rows(args.right), json.loads(args.plan.read_text())['rows']['internal'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
