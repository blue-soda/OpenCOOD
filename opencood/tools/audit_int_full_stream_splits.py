"""Inventory single-car DAIR and V2V4Real splits before full training, without opening target boxes."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re

from opencood.data_utils.int_ego_stream import build_stream_manifest


def v2v_manifest(root):
    rows, issues = [], []
    for split in ('train', 'val', 'test'):
        containers = sorted(p for p in (root / split).rglob('0') if p.is_dir() and (p.parent / '1').is_dir())
        if not containers:
            raise ValueError(('no paired drives', split))
        for ego in containers:
            points = {p.stem:p for p in ego.glob('*.pcd')}
            labels = {p.stem:p for p in ego.glob('*.yaml') if p.stem.isdigit()}
            if points.keys() != labels.keys():
                issues.append(dict(scene=ego.parent.name, split=split, point_only=sorted(points.keys()-labels.keys()),
                                   yaml_only=sorted(labels.keys()-points.keys())))
            for frame in sorted(points.keys() & labels.keys()):
                rows.append(dict(split=split, scene=ego.parent.name, drive=re.sub(r'_\d+$', '', ego.parent.name),
                                 frame=frame, pcd=str(points[frame]), annotation=str(labels[frame])))
    sizes = defaultdict(list)
    for row in rows:
        sizes[Path(row['pcd']).stat().st_size].append(row)
    exact = []
    for group in sizes.values():
        if len({r['split'] for r in group}) < 2:
            continue
        by_hash = defaultdict(list)
        for r in group:
            by_hash[hashlib.sha256(Path(r['pcd']).read_bytes()).hexdigest()].append(r)
        for key, members in by_hash.items():
            if len({r['split'] for r in members}) > 1:
                exact.append(dict(sha256=key, locations=[[r['split'],r['scene'],r['frame']] for r in members]))
    overview = {}
    for split in ('train','val','test'):
        selected = [r for r in rows if r['split'] == split]
        overview[split] = dict(scans=len(selected), scenes=len({r['scene'] for r in selected}),
                               drives=len({r['drive'] for r in selected}))
    overlap = {}
    for left,right in itertools.combinations(overview, 2):
        overlap[left+'_'+right] = {field:sorted({r[field] for r in rows if r['split']==left} &
                                                {r[field] for r in rows if r['split']==right})
                                   for field in ('scene','drive')}
    return rows, dict(splits=overview, missing_pairs=issues, intersections=overlap, duplicate_clouds=exact,
                      ego='fixed agent 0; file pairing only; object-label semantics and visibility not validated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    v2v,report = v2v_manifest(args.datasets/'v2v4real/extracted')
    (args.out/'v2v_summary.json').write_text(json.dumps(report, indent=2))
    with (args.out/'v2v_rows.jsonl').open('x') as stream:
        for row in v2v:
            stream.write(json.dumps(row)+'\n')
    dair = build_stream_manifest(args.datasets/'dair-v2x-c/extracted/cooperative-vehicle-infrastructure')
    summary = {}
    for split in ('train','val'):
        records = dair[split]
        summary[split] = dict(scans=len(records), labels=sum(bool(r['supervised']) for r in records),
                              scenes=len({r['scene'] for r in records}), segments=len({r['segment'] for r in records}))
    summary['overlap'] = sorted({r['scene'] for r in dair['train']} & {r['scene'] for r in dair['val']})
    (args.out/'dair_summary.json').write_text(json.dumps(summary, indent=2))
    (args.out/'dair_manifest.json').write_text(json.dumps(dair))
