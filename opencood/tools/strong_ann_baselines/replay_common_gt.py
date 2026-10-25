"""Re-score exported ANN predictions with our exact GT and evaluator; never changes predictions."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--root',default='/data1/zhh/gzc/diagnostics/ann_baselines_20261025');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
sys.path.insert(0,'/data1/zhh/gzc/OpenCOOD')
from opencood.utils import eval_utils
run=Path(a.run);root=Path(a.root);ids=json.loads((root/'common_val.json').read_text())
paths=sorted((run/('frames' if (run/'frames').is_dir() else 'predictions')).glob('*.npz'))
if not a.allow_partial:
    assert (run/'summary.json').exists() and len(paths)==len(ids),'Full run not completed'
stat={t:{'tp':[],'fp':[],'gt':0,'score':[]} for t in (.3,.5,.7)}
original={t:{'tp':[],'fp':[],'gt':0,'score':[]} for t in stat}
rows=[]
manifest=run/'frames.jsonl'
index=[json.loads(s) for s in manifest.read_text().splitlines()] if manifest.exists() else json.loads((run/'predictions_index.json').read_text())
indexed={x['file']:x for x in index}
assert len(indexed)==len(paths), 'Prediction manifest count mismatch'
gt_index={x['frame_id']:x for x in json.loads((root/'common_gt_manifest.json').read_text())['frames']}
for i,path in enumerate(paths):
    entry=indexed[str(path.relative_to(run))]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'], 'Prediction SHA mismatch'
    z=np.load(str(path));fid=str(z['frame_id'].item());assert fid==ids[i]
    assert fid==entry['frame_id']
    gt_path=root/'common_gt'/(fid+'.npy')
    assert hashlib.sha256(gt_path.read_bytes()).hexdigest()==gt_index[fid]['sha256'], 'GT SHA mismatch'
    gt=np.load(str(gt_path));official=z['gt']
    # Match unordered boxes by BEV corner coordinates; z differences are also recorded.
    matched=False;xymax=None;xyzmax=None
    if len(gt)==len(official):
        if len(gt):
            cost=np.abs(gt[:,None,:,:2]-official[None,:,:,:2]).max(axis=(2,3))
            x,y=linear_sum_assignment(cost);xymax=float(cost[x,y].max());xyzmax=float(np.abs(gt[x]-official[y]).max());matched=xymax<1e-3
        else:matched=True
    for t in stat:
        pred=torch.from_numpy(z['pred']);score=torch.from_numpy(z['score'])
        eval_utils.caluclate_tp_fp(pred,score,torch.from_numpy(gt),stat,t)
        eval_utils.caluclate_tp_fp(pred,score,torch.from_numpy(official),original,t)
    rows.append({'frame_id':fid,'our_gt':len(gt),'official_gt':len(official),'bev_gt_match':matched,'max_xy_error':xymax,'max_xyz_error':xyzmax})
report={'samples':len(paths),'full_split':len(paths)==len(ids),'not_a_main_result':len(paths)!=len(ids),'bev_gt_mismatch_frames':sum(not x['bev_gt_match'] for x in rows),'our_gt_boxes':stat[.7]['gt'],'official_gt_boxes':original[.7]['gt'],'common_gt_bev_ap':{str(t):float(eval_utils.calculate_ap(stat,t)[0]) for t in stat},'original_gt_our_evaluator_bev_ap':{str(t):float(eval_utils.calculate_ap(original,t)[0]) for t in stat},'frames':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
report['prediction_sha_verified']=len(paths)
report['gt_sha_verified']=len(paths)
target=run/('common_gt_replay_partial.json' if a.allow_partial else 'common_gt_replay.json');target.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))
