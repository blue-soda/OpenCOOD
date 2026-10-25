import json, hashlib, zipfile, shutil
from pathlib import Path
base=Path('/data1/zhh/gzc')
out=base/'diagnostics/ann_baselines_20261025'
root=base/'dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure'
manifest=json.loads((base/'diagnostics/e3dsnn_event_utility_source_20261022/pair_manifest.json').read_text())
report={'splits':{},'backup_restore':{},'extensions':{}}
for split in ('train','val'):
    ids=manifest['splits'][split]['valid_ids']
    p=out/('common_'+split+'.json');p.write_text(json.dumps(ids,indent=2))
    report['splits'][split]={'count':len(ids),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
archive=base/'dataset/dair-v2x-c/cooperative-vehicle-infrastructure.zip'
report['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
with zipfile.ZipFile(str(archive)) as z:
    for src,dst in [('cooperative/label_world','cooperative/label_world_backup'),('vehicle-side/label/lidar','vehicle-side/label/lidar_backup')]:
        prefix='cooperative-vehicle-infrastructure/'+src+'/'
        target=root/dst;target.mkdir(exist_ok=True)
        n=0
        for name in z.namelist():
            if not name.startswith(prefix) or not name.endswith('.json'):continue
            leaf=name[len(prefix):]
            if '/' in leaf or '\\' in leaf:raise ValueError(name)
            data=z.read(name);p=target/leaf
            if p.exists():assert p.read_bytes()==data
            else:p.write_bytes(data)
            n+=1
        report['backup_restore'][dst]=n
source=base/'OpenCOOD/opencood/pcdet_utils/iou3d_nms'
for repo in ('CoSDH','InfoCom'):
    dest=base/repo/'opencood/pcdet_utils/iou3d_nms'
    srcs=list((dest/'src').glob('*'))
    same=bool(srcs) and all((source/'src'/p.name).read_bytes()==p.read_bytes() for p in srcs)
    report['extensions'][repo]={'source_byte_identical':same}
    if same:
        for p in source.glob('*.so'):
            shutil.copy2(str(p),str(dest/p.name))
            report['extensions'][repo]['binary_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
