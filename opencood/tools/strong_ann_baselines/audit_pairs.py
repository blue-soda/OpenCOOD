import json,sys,numpy as np
from pathlib import Path
sys.path.insert(0,'/data1/zhh/gzc/OpenCOOD')
from opencood.data_utils.datasets.paired_dair_fusion import load_pair_geometry
r=Path('/data1/zhh/gzc/dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure');out=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');ids=json.loads((out/'common_val.json').read_text())
info={Path(x['vehicle_pointcloud_path']).stem:x for x in json.loads((r/'cooperative/data_info.json').read_text())}
rows=[{'index':i,'frame_id':f,'distance_xy':float(np.linalg.norm(load_pair_geometry(r,info[f])[1][:2,3]))} for i,f in enumerate(ids)]
near=[x for x in rows if x['distance_xy']<90]
(out/'cooperative_smoke_ids.json').write_text(json.dumps([x['frame_id'] for x in near[:2]]))
(out/'pair_distance_audit.json').write_text(json.dumps({'over_100m':sum(x['distance_xy']>100 for x in rows),'frames':rows},indent=2))
print(json.dumps({'first':rows[:2],'near':near[:2],'over_100m':sum(x['distance_xy']>100 for x in rows)}))
