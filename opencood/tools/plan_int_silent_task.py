"""Freeze scene-disjoint adapter fit and internal evaluation from the original train split."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(base, vehicle, output):
    native_path = base / 'r328_manifest.json'
    original_path = base / 'supervised_trace/plan.json'
    original = json.loads(original_path.read_bytes())
    manifest = json.loads(native_path.read_bytes())
    assert original['manifest'] == digest(native_path)
    known = {r['scene'] for r in original['rows']}
    validation = {r['scene'] for r in manifest['val']}
    sealed = {'2', '31', '50', '121'}
    segments = defaultdict(lambda: defaultdict(list))
    for row in manifest['train']:
        assert row['split'] == 'train' and row['scene'] not in validation | sealed
        segments[row['scene']][row['segment']].append(row)
    assert len(segments) == 52 and len(known) == 8
    choices = {}
    for scene, groups in segments.items():
        candidates = []
        for segment, rows in groups.items():
            prefix = rows[:64]
            assert all(int(a['timestamp_us'])<int(b['timestamp_us']) for a,b in zip(prefix,prefix[1:]))
            if sum(bool(r['supervised']) for r in prefix)>=4:
                candidates.append((-len(prefix), segment, prefix))
        if candidates:
            choices[scene] = min(candidates)[2]
    assert known <= choices.keys()
    rank = lambda s: hashlib.sha256(('silent-task/303/'+s).encode()).digest()
    ordered = sorted(choices.keys()-known, key=rank)
    assert len(ordered)>=28
    splits = {'fit': sorted(known, key=rank)+ordered[:16], 'internal': ordered[16:28]}
    report = dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                  train_manifest=digest(native_path), short_plan=digest(original_path),seed=303,
                  original_fit_scenes=sorted(known),eligible_scenes=len(choices),splits={})
    full = {'members':{}, 'rows':{}}
    for key, scenes in splits.items():
        selected = []
        for scene in scenes:
            for row in choices[scene]:
                r = dict(row)
                for field, folder in [('pcd','velodyne'),('label','label/lidar')]:
                    if r[field]:
                        path = vehicle / folder / Path(r[field]).name
                        full['members'][str(path)] = digest(path)
                        r[field] = str(path)
                selected.append(r)
        full['rows'][key] = selected
        report['splits'][key] = dict(scenes=scenes, scans=len(selected),
                                     labels=sum(bool(r['supervised']) for r in selected))
    assert not (set(splits['fit']) & set(splits['internal']))
    paths = [{r['pcd'] for r in rs} for rs in full['rows'].values()]
    assert not (paths[0]&paths[1])
    hashes = [{full['members'][p] for p in files} for files in paths]
    assert not (hashes[0]&hashes[1])
    output.mkdir(parents=True,exist_ok=False)
    target = output / 'plan.json'
    target.write_text(json.dumps(full,indent=2))
    report.update(status='COMPLETED',plan=digest(target),optimizer_steps=0,model_forwards=0)
    (output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('long-base','vehicle','output'):
        p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    run(args.long_base,args.vehicle,args.output)
