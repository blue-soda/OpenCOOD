"""Check strict checkpoint reload, batched forward and nonempty 7D decoding."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = load_yaml('opencood/hypes_yaml/dair-v2x/snn/e3dsnn_vehicle_single.yaml')
    cfg['data_augment'] = []
    dataset = build_dataset(cfg, train=True)
    np.random.seed(20260929)
    samples = [dataset[i] for i in range(4)]
    assert all(set(sample) == {'ego'} for sample in samples)
    model = train_utils.create_model(cfg).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    batch = train_utils.to_device(dataset.collate_batch_train(samples), torch.device('cuda'))
    model.eval()
    with torch.no_grad():
        result = model(batch['ego'])
        assert list(result['psm'].shape) == [4, 2, 50, 126]
        assert all(torch.isfinite(result[key]).all() for key in ('psm', 'rm', 'dm'))
        boxes = dataset.post_processor.delta_to_boxes3d(result['rm'], batch['ego']['anchor_box'])
        assert list(boxes.shape) == [4, 12600, 7]
        assert torch.isfinite(boxes).all() and (boxes[:, :, 3:6] > 0).all()
    report = {'strict_reload': True, 'vehicle_only': True, 'frame_ids': dataset.data[:4],
              'prediction_shape': list(result['psm'].shape), 'raw_box_shape': list(boxes.shape),
              'finite_positive_box_sizes': True, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
