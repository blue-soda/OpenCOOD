"""Decode every scan before freezing an INT stream; split at invalid scans."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from opencood.data_utils.int_ego_stream import sha256
from opencood.utils import pcd_utils


def check(row):
    points, _ = pcd_utils.read_pcd(row['pcd'])
    reason = None
    if len(points) == 0:
        reason = 'decoded_empty_pcd'
    elif not np.isfinite(points).all():
        reason = 'nonfinite_decoded_points'
    result = dict(frame=row['frame'], split=row['split'], points=len(points),
                  bytes=Path(row['pcd']).stat().st_size, reason=reason)
    if reason:
        result['sha256'] = sha256(row['pcd'])
    return result


def filter_stream(rows, audits):
    if len(rows) != len(audits):
        raise ValueError('Incomplete decode audit')
    output, counts = [], {}
    for row, audit in zip(rows, audits):
        if row['frame'] != audit['frame']:
            raise ValueError('Audit order mismatch')
        old = row['segment']
        if audit['reason']:
            counts[old] = counts.get(old, 0)+1
            continue
        new = dict(row)
        if counts.get(old, 0):
            new['segment'] = old+':decode%d' % counts[old]
        output.append(new)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    m = json.loads(Path(a.manifest).read_text())
    for split in ['train', 'val']:
        audits = []
        with ProcessPoolExecutor(max_workers=a.workers) as pool, (root/(split+'_decode.jsonl')).open('w', buffering=1) as log:
            for audit in pool.map(check, m[split], chunksize=16):
                audits.append(audit)
                log.write(json.dumps(audit)+'\n')
                if len(audits) % 1000 == 0:
                    print(split, 'decoded', len(audits), flush=True)
        m[split] = filter_stream(m[split], audits)
        m['excluded'].extend(x for x in audits if x['reason'])
        m['summary'][split] = dict(scans=len(m[split]), labels=sum(x['supervised'] for x in m[split]),
             scenes=len(set(x['scene'] for x in m[split])), segments=len(set(x['segment'] for x in m[split])))
    m['protocol']['decode_audit'] = dict(parent_sha256=sha256(a.manifest),
              rule='All scans decoded with training reader; exclude empty/nonfinite scans and split state continuity')
    (root/'sequence_manifest.json').write_text(json.dumps(m, indent=2))
    print(json.dumps(m['summary']), flush=True)


if __name__ == '__main__':
    main()
