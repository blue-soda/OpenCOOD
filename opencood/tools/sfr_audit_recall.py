"""Dev-only bottleneck audit. GT and current infra are diagnostic labels only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd())
from opencood.tools.sfr_run import seed, digest, plain, prepare
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.sfr_dair import SfrDAIRDataset
from opencood.models.sfr.sender import SfrSender
from opencood.models.sfr.temporal import Temporal
from opencood.models.sfr.pillar_fusion import SfrPillarFusion
from opencood.tools.train_utils import to_device
from opencood.utils import box_utils, common_utils, eval_utils


def gt_max_iou(corners, gt):
    """Independent per-GT coverage; not greedy detection recall or AP."""
    if corners is None or not len(corners):
        return np.zeros(len(gt))
    polygons = list(common_utils.convert_format(corners.detach().cpu().numpy()))
    targets = common_utils.convert_format(gt.detach().cpu().numpy())
    return np.array([common_utils.compute_iou(g, polygons).max() for g in targets])


def mask_at_centers(mask, gt, bounds):
    centers = gt.mean(1)
    h, w = mask.shape[-2:]
    x = ((centers[:, 0]-bounds[0])*w/(bounds[3]-bounds[0])).floor().long()
    y = ((centers[:, 1]-bounds[1])*h/(bounds[4]-bounds[1])).floor().long()
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    values = mask.new_zeros(len(gt))
    values[valid] = mask[0, 0, y[valid], x[valid]]
    return values.cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split_file', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--p', type=float, default=.3)
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg['sender'].get('feature_level') != 'pillar':
        raise ValueError('This audit requires the pillar decoder')
    base = load_yaml(cfg['base_config'])
    base['validate_dir'] = args.split_file
    base['binomial_p'] = args.p
    base['ectra_ego_history'] = dict(enabled=True, max_age_ms=cfg['history_max_age_ms'])
    seed(303)
    dataset = SfrDAIRDataset(copy.deepcopy(base), False, False)
    device = torch.device('cuda')
    sender = SfrSender(base['model']['args'], cfg['sender']).to(device)
    sender.load_state_dict(torch.load(cfg['single_checkpoint'], map_location='cpu'), strict=True)
    temporal = Temporal(cfg['temporal']).to(device).eval()
    model = SfrPillarFusion(sender, cfg['fusion']).to(device).eval()
    saved = torch.load(args.checkpoint, map_location='cpu')
    if saved['config'] != cfg or saved['stage'] != 'fusion':
        raise ValueError('Checkpoint/config mismatch')
    model.load_state_dict(saved['model'], strict=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=dataset.collate_batch_test, generator=torch.Generator().manual_seed(303))
    manifest = dict(arguments=vars(args), checkpoint_sha256=digest(args.checkpoint),
                    single_sha256=digest(cfg['single_checkpoint']), split_sha256=digest(args.split_file),
                    source_sha256={str(f):digest(f) for f in list(Path('opencood/models/sfr').glob('*.py'))+
                                   [Path(__file__),Path('opencood/tools/sfr_run.py'),Path('opencood/data_utils/datasets/sfr_dair.py')]},
                    config=cfg, note='GT and current-infra pseudo boxes only measure coverage; never enter messages or predictions. '
                    'binary_prediction_support is a frozen-checkpoint diagnostic intervention, not a matched trained ablation.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    names = ['ego', 'v4_soft', 'binary_prediction_support']
    stats = {n:{t:dict(tp=[], fp=[], gt=0, score=[]) for t in (.3,.5,.7)} for n in names}
    all_iou, all_alpha, identities = {}, [], []
    for index, batch in enumerate(loader):
        if args.max_steps and index >= args.max_steps:break
        if batch is None:continue
        batch = to_device(batch, device)
        with torch.no_grad():
            ego, messages, target, _ = prepare(batch, sender, dataset, cfg, need_target=True)
            tracks, _ = temporal.sequence(messages, mode='cv')
            transported = temporal.collect(tracks, include_orientation=cfg['fusion'].get('orientation_conditioned', False))
            output, diagnostic = model(ego, transported, return_components=True)
            baseline, decoded, mask = diagnostic.pop('components')
            binary = {key:baseline[key]+(mask>0).to(mask)*(value-baseline[key]) for key,value in decoded.items()}
            gt = dataset.post_processor.generate_gt_bbx(batch)
            ious = {}
            for name, prediction in zip(names, (baseline, output, binary)):
                corners, scores = dataset.post_processor.post_process(batch, {'ego':prediction})
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(corners, scores, gt, stats[name], threshold)
                ious[name] = gt_max_iou(corners, gt)
            infra = [m for m in messages if m['metadata']['agent_id'] != '0']
            latest = max(infra, key=lambda m:m['metadata']['time_s'])['boxes']
            predicted = torch.stack([t.box for t in tracks if '1' in t.caches]) if any('1' in t.caches for t in tracks) else latest[:0]
            for name, boxes in [('infra_latest_candidates',latest), ('joint_track_candidates',predicted),
                                ('current_infra_candidates_diagnostic_only',target['boxes'])]:
                ious[name] = gt_max_iou(box_utils.boxes_to_corners_3d(boxes, order='hwl'), gt)
            alpha = mask_at_centers(mask, gt, sender.bounds)
            for name, values in ious.items():all_iou.setdefault(name, []).extend(values.tolist())
            all_alpha.extend(alpha.tolist())
        meta = plain(batch['ego']['sfr_metadata'][0])
        identity = dict(frame_id=meta['vehicle_frame'], intervals=plain(batch['ego']['past_k_time_interval']),
                        timestamps=[[m['observation_us'] for m in row] for row in meta['observations']])
        identities.append(identity)
        row = dict(index=index, sample=identity, gt=len(gt), iou={n:v.tolist() for n,v in ious.items()},
                   support_alpha_at_gt=alpha.tolist(), diagnostics=diagnostic)
        with (out/'per_gt.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if index%100 == 0:print(json.dumps(dict(index=index, gt=len(gt), diagnostics=diagnostic)), flush=True)
    arrays = {n:np.array(v) for n,v in all_iou.items()}
    alpha = np.array(all_alpha)
    groups = dict(all=np.ones(len(alpha),dtype=bool), ego_missed_iou50=arrays['ego']<.5,
                  ego_missed_iou70=arrays['ego']<.7)
    report = dict(samples=len(identities), gt_count=len(alpha), smoke=bool(args.max_steps),
                  sample_time_sha256=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(), arms={}, groups={})
    for name in names:
        path=out/name;path.mkdir()
        report['arms'][name]=dict(zip(('ap30','ap50','ap70'),eval_utils.eval_final_results(stats[name],str(path),dataset='d')))
    for name, select in groups.items():
        report['groups'][name] = dict(count=int(select.sum()),
            coverage_iou50={n:float((v[select]>=.5).mean()) for n,v in arrays.items()},
            coverage_iou70={n:float((v[select]>=.7).mean()) for n,v in arrays.items()},
            support_nonzero=float((alpha[select]>0).mean()), support_mean=float(alpha[select].mean()),
            support_quantiles=np.quantile(alpha[select],[0,.25,.5,.75,1]).tolist()) if select.any() else dict(count=0)
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
