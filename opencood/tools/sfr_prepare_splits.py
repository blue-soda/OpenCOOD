"""Create deterministic scene-disjoint SFR development splits from official train only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=2301)
    parser.add_argument('--development_fraction', type=float, default=.2)
    args = parser.parse_args()
    root, out = Path(args.data_root), Path(args.output)
    vehicle = {Path(x['pointcloud_path']).stem: x for x in json.loads((root/'vehicle-side/data_info.json').read_text()) if x['pointcloud_path']}
    infra = {Path(x['pointcloud_path']).stem: x for x in json.loads((root/'infrastructure-side/data_info.json').read_text()) if x['pointcloud_path']}
    rows = json.loads((root/'cooperative/data_info.json').read_text())
    train, official_val = json.loads((root/'train.json').read_text()), json.loads((root/'val.json').read_text())
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    missing_metadata = []
    for row in rows:
        a, b = Path(row['vehicle_pointcloud_path']).stem, Path(row['infrastructure_pointcloud_path']).stem
        if a not in vehicle or b not in infra:
            missing_metadata.append([a, b])
            continue
        a, b = 'v'+str(vehicle[a]['batch_id']), 'i'+str(infra[b]['batch_id'])
        parents[find(a)] = find(b)
    groups = defaultdict(list)
    for fid in train:
        groups[find('v'+str(vehicle[fid]['batch_id']))].append(fid)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    development, chosen = set(), []
    for group in keys:
        if len(development) >= len(train)*args.development_fraction:
            break
        development.update(groups[group])
        chosen.append(group)
    fit = [fid for fid in train if fid not in development]
    dev = [fid for fid in train if fid in development]
    if not fit or not dev:
        raise ValueError('Insufficient connected scene groups for a disjoint split')
    assert not set(fit)&set(dev) and not set(train)&set(official_val)
    out.mkdir(parents=True, exist_ok=False)
    (out/'fit.json').write_text(json.dumps(fit))
    (out/'dev.json').write_text(json.dumps(dev))
    metadata = dict(seed=args.seed, source=str(root/'train.json'),
                    source_sha256=hashlib.sha256((root/'train.json').read_bytes()).hexdigest(),
                    fit_count=len(fit), dev_count=len(dev), groups={key: len(value) for key, value in groups.items()},
                    dev_groups=chosen, excluded_invalid_cooperative_metadata=missing_metadata,
                    note='No original val frames used in development. Connected vehicle/infrastructure batch groups stay together; sampler unchanged.')
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
