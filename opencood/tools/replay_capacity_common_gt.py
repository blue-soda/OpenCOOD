"""Re-score immutable SNN predictions with the exact strong-ANN CPU GT."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from opencood.utils import eval_utils,eval_3d_utils

GT_SHA='77657e251057b5323a97b4b8f0820ec2e1dfa3b4b745e87c6f13902816a904ba'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def replay(folder,reference,budget):
    e=read(folder/'evaluation.json');native=read(folder/'offline_replay.json')
    assert native['passed'] and native['exact_match'] and native['all_file_hashes_verified']
    assert native['ap']==e['ap'] and native['ap_3d']==e['ap_3d']
    index=read(folder/'predictions/index.json')['frames']
    assert sha(folder/'predictions/index.json')==e['predictions']['index_sha256']
    refs=read(reference/'common_gt_manifest.json')['frames']
    ids=read(reference/'common_val.json')
    packets=[json.loads(s) for s in (folder/'packets/index.jsonl').read_text().splitlines()]
    assert len(index)==len(refs)==len(ids)==len(packets)==e['samples']==1738 and e['gt_boxes']==28622
    assert len(set(ids))==len(ids)
    stat={t:{'tp':[],'fp':[],'gt':0,'score':[]} for t in (.3,.5,.7)}
    stat3={t:{'tp':[],'fp':[],'gt':0,'score':[]} for t in stat}
    digest=hashlib.sha256();differences=[];comm=[];total_gt=0
    for position,(row,ref,packet,fid) in enumerate(zip(index,refs,packets,ids)):
        assert row['index']==position and row['frame_id']==ref['frame_id']==packet['frame_id']==fid
        p=folder/'predictions'/row['file'];gpath=reference/'common_gt'/(fid+'.npy');wire=folder/'packets'/packet['file']
        assert p.resolve().parent==(folder/'predictions').resolve()
        assert wire.resolve().parent==(folder/'packets').resolve()
        assert sha(p)==row['sha256'] and sha(gpath)==ref['sha256'] and sha(wire)==packet['sha256']
        size=wire.stat().st_size
        assert size==packet['actual_bytes']==row['communication']['total_bytes'] and 0<=size<=budget
        with np.load(str(p),allow_pickle=False) as z:
            assert str(z['frame_id'].item())==fid
            boxes=z['pred_boxes'].copy();scores=z['scores'].copy();original=z['gt_boxes'].copy()
        gt=np.load(str(gpath),allow_pickle=False)
        assert boxes.shape==(row['predicted_boxes'],8,3) and scores.shape==(len(boxes),)
        assert gt.shape==original.shape==(row['gt_boxes'],8,3)
        assert all(np.isfinite(v).all() for v in (boxes,scores,original,gt))
        digest.update(fid.encode());digest.update(str(gt.shape).encode());digest.update(gt.tobytes())
        if not np.array_equal(gt,original):
            cost=np.abs(original[:,None]-gt[None,:]).max(axis=(2,3));ii,jj=linear_sum_assignment(cost)
            differences.append(dict(frame_id=fid,max_matched_xyz=float(cost[ii,jj].max()) if len(gt) else 0.))
        for t in stat:
            eval_utils.caluclate_tp_fp(torch.from_numpy(boxes),torch.from_numpy(scores),torch.from_numpy(gt),stat,t)
        eval_3d_utils.calculate_tp_fp_3d(torch.from_numpy(boxes),torch.from_numpy(scores),torch.from_numpy(gt),stat3)
        total_gt+=len(gt);comm.append(dict(frame_id=fid,bytes=size))
    assert digest.hexdigest()==GT_SHA and total_gt==28622
    assert sum(x['bytes'] for x in comm)==e['feature_packets']['total_bytes']
    result=dict(passed=True,samples=1738,gt_boxes=total_gt,full_split=True,common_gt_sha256=GT_SHA,
        ap={str(t):eval_utils.calculate_ap(stat,t)[0] for t in stat},
        ap_3d={str(t):eval_utils.calculate_ap(stat3,t)[0] for t in stat3},
        original_ap=e['ap'],original_ap_3d=e['ap_3d'],feature_packets=e['feature_packets'],
        prediction_index_sha256=sha(folder/'predictions/index.json'),packet_index_sha256=sha(folder/'packets/index.jsonl'),
        reference_manifest_sha256=sha(reference/'common_gt_manifest.json'),reference=str(reference),
        unchanged_predictions=True,all_prediction_and_packet_hashes_verified=True,
        gt_order_or_geometry_differences=differences,communication_frames=comm)
    target=folder/'common_gt_evaluation.json'
    if target.exists():assert read(target)==result,'Existing common-GT result differs'
    else:target.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation-dir',required=True);p.add_argument('--reference',required=True);p.add_argument('--budget',type=int,required=True)
    a=p.parse_args();r=replay(Path(a.evaluation_dir),Path(a.reference),a.budget)
    print(json.dumps({k:v for k,v in r.items() if k not in ('communication_frames','gt_order_or_geometry_differences')}))
