import tarfile,json,time
from pathlib import Path
r=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025')
(r/'snapshot_time.json').write_text(json.dumps({'unix':time.time()}))
with tarfile.open(str(r/'evidence_snapshot.tar.gz'),'w:gz') as t:
    for p in sorted(r.rglob('*.json')):
        if len(p.relative_to(r).parts)<=2:t.add(str(p),arcname=str(p.relative_to(r)))
    t.add(str(r/'requirements_resolved.txt'),arcname='requirements_resolved.txt')
print((r/'evidence_snapshot.tar.gz').stat().st_size)
