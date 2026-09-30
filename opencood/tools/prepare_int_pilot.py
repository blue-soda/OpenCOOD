"""Select small complete segments for engineering tests, never scientific AP."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path


def select(rows, count):
    groups = OrderedDict()
    for row in rows:
        groups.setdefault(row['segment'], []).append(row)
    chosen = [group for group in groups.values()
              if 8 <= len(group) <= 24 and any(r['supervised'] for r in group)
              and any(not r['supervised'] for r in group)][:count]
    if len(chosen) != count:
        raise ValueError('Not enough labelled/unlabelled complete short segments')
    return [dict(row) for group in chosen for row in group]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.manifest).read_bytes()
    manifest = json.loads(source)
    result = {split: select(manifest[split], 2) for split in ('train', 'val')}
    result['protocol'] = dict(purpose='engineering-only two-epoch TBPTT/coverage/resume pilot; AP not a research result',
                              parent_manifest_sha256=hashlib.sha256(source).hexdigest(),
                              parent_manifest=str(Path(args.manifest).resolve()))
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
    print({split: dict(scans=len(result[split]), labels=sum(r['supervised'] for r in result[split]))
           for split in ('train', 'val')})


if __name__ == '__main__':
    main()
