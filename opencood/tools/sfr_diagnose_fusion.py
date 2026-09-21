"""Same-sample diagnostic controls; dense communication is explicitly not SFR."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
import yaml
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd())
from opencood.tools.sfr_run import seed, digest, plain, prepare
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.sfr_dair import SfrDAIRDataset
from opencood.models.sfr.sender import SfrSender
from opencood.models.sfr.temporal import Temporal
from opencood.models.sfr.geometry import warp_map, reader_to_reference
from opencood.tools.train_utils import to_device
from opencood.utils import box_utils, eval_utils


def late_predictions(boxes, scores, processor):
    corners = box_utils.boxes_to_corners_3d(boxes, order='hwl')
    if len(corners):
        keep = box_utils.remove_large_pred_bbx(corners) & box_utils.remove_bbx_abnormal_z(corners)
        corners, scores = corners[keep], scores[keep]
        keep = box_utils.nms_rotated(corners, scores, processor.params['nms_thresh'])
        corners, scores = corners[keep], scores[keep]
        keep = box_utils.get_mask_for_boxes_within_range_torch(corners, processor.params['gt_range'])
        corners, scores = corners[keep], scores[keep]
    return corners, scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--split_file', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--p', type=float, default=.3)
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(Path(args.config).read_text())
    base = load_yaml(cfg['base_config'])
    base['validate_dir'] = args.split_file
    base['binomial_p'] = args.p
    base['ectra_ego_history'] = dict(enabled=True, max_age_ms=100)
    seed(303)
    data = SfrDAIRDataset(copy.deepcopy(base), False, False)
    device = torch.device('cuda')
    sender = SfrSender(base['model']['args'], cfg['sender']).to(device)
    sender.load_state_dict(torch.load(cfg['single_checkpoint'], map_location='cpu'), strict=True)
    temporal = Temporal(cfg['temporal']).to(device).eval()
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=data.collate_batch_test, generator=torch.Generator().manual_seed(303))
    names = ['ego', 'late_latest', 'late_cv', 'dense_max_untrained']
    statistics = {name: {x: dict(tp=[], fp=[], gt=0, score=[]) for x in (.3, .5, .7)} for name in names}
    metadata = dict(arguments=vars(args), single_sha256=digest(cfg['single_checkpoint']),
                    split_sha256=digest(args.split_file), config=cfg,
                    source_sha256={str(p):digest(p) for p in list(Path('opencood/models/sfr').glob('*.py'))+
                                   [Path(__file__),Path('opencood/tools/sfr_run.py'),Path('opencood/data_utils/datasets/sfr_dair.py')]},
                    note='Diagnostic only: late boxes and dense_max are not claimed SFR models')
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2))
    identities = []
    for index, batch in enumerate(loader):
        if args.max_steps and index >= args.max_steps:
            break
        if batch is None:
            continue
        batch = to_device(batch, device)
        with torch.no_grad():
            ego, messages, _, _ = prepare(batch, sender, data, cfg)
            k = batch['ego']['past_k_time_interval'].numel()//2
            dense, _ = sender.extract(batch['ego']['processed_lidar'], 2*k, batch['ego']['anchor_box'], data.post_processor)
            def head(feature):
                result = dict(psm=sender.cls_head(feature), rm=sender.reg_head(feature))
                if sender.use_dir: result['dm'] = sender.dir_head(feature)
                return data.post_processor.post_process(batch, dict(ego=result))
            predictions = dict(ego=head(ego))
            current = [m for m in messages if m['metadata']['agent_id'] == '0'][0]
            infra = max([m for m in messages if m['metadata']['agent_id'] == '1'], key=lambda m:m['metadata']['time_s'])
            predictions['late_latest'] = late_predictions(torch.cat((current['boxes'], infra['boxes'])), torch.cat((current['scores'], infra['scores'])), data.post_processor)
            tracks, _ = temporal.sequence([m for m in messages if m['metadata']['agent_id'] == '1'], mode='cv')
            boxes = torch.stack([t.box for t in tracks]) if tracks else current['boxes'][:0]
            scores = torch.stack([t.score for t in tracks]) if tracks else current['scores'][:0]
            predictions['late_cv'] = late_predictions(torch.cat((current['boxes'], boxes)), torch.cat((current['scores'], scores)), data.post_processor)
            warped = warp_map(dense[k:k+1], reader_to_reference(batch['ego']['pairwise_t_matrix'][0,1,0].to(dense)), sender.bounds)
            predictions['dense_max_untrained'] = head(torch.maximum(ego, warped))
            gt = data.post_processor.generate_gt_bbx(batch)
            for name, (corners, scores) in predictions.items():
                for threshold in statistics[name]:
                    eval_utils.caluclate_tp_fp(corners, scores, gt, statistics[name], threshold)
        meta = plain(batch['ego']['sfr_metadata'][0])
        identities.append(dict(frame_id=meta['vehicle_frame'], intervals=plain(batch['ego']['past_k_time_interval']),
                               timestamps=[[m['observation_us'] for m in row] for row in meta['observations']]))
        if index % 100 == 0: print(json.dumps(dict(index=index, counts={n:len(v[1]) if v[1] is not None else 0 for n,v in predictions.items()}, gt=len(gt))), flush=True)
    result = dict(samples=len(identities), smoke=bool(args.max_steps), sample_time_sha256=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(), arms={})
    for name in names:
        path=out/name; path.mkdir()
        ap=eval_utils.eval_final_results(statistics[name],str(path),dataset='d')
        result['arms'][name]=dict(zip(('ap30','ap50','ap70'),ap))
    (out/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
