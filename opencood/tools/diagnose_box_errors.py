"""Report nearest-object box errors without changing the evaluation protocol."""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import box_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch_stats', action='store_true')
    args = parser.parse_args()
    config = yaml_utils.load_yaml(None, args)
    config['validate_dir'] = config['test_dir']
    config['binomial_p'] = 0.
    np.random.seed(303)
    dataset = build_dataset(config, visualize=False, train=False)
    model = train_utils.create_model(config).cuda().eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    else:
        train_utils.load_saved_model_diff(args.model_dir, model)
    if args.batch_stats:
        model.train()
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test)
    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.samples:
                break
            if batch is None:
                continue
            batch = train_utils.to_device(batch, torch.device('cuda'))
            out = model(batch['ego'], dataset) if args.two_stage else model(batch['ego'])
            pred, score, gt = dataset.post_process(batch, {'ego': out})
            if pred is None or gt is None or not len(pred) or not len(gt):
                continue
            pred = box_utils.corner_to_center(pred.cpu().numpy(), order='hwl')
            gt = box_utils.corner_to_center(gt.cpu().numpy(), order='hwl')
            distance = np.linalg.norm(pred[:, None, :2] - gt[None, :, :2], axis=-1)
            match = distance.argmin(axis=1)
            keep = distance.min(axis=1) < 1.
            residual = pred[keep] - gt[match[keep]]
            residual[:, 6] = (residual[:, 6] + np.pi / 2) % np.pi - np.pi / 2
            rows.extend(residual.tolist())
    if not rows:
        print(json.dumps({'matches': 0, 'reason': 'No predicted center within 1m of a GT center'}))
        return
    errors = np.asarray(rows)
    print(json.dumps({'matches': len(rows), 'coordinate_order': 'x,y,z,h,w,l,yaw',
                      'median_signed': np.median(errors, axis=0).tolist(),
                      'median_abs': np.median(abs(errors), axis=0).tolist()}, indent=2))


if __name__ == '__main__':
    main()
