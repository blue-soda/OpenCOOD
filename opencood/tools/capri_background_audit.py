"""Isolated sparse/dense ego-reference audit; does not change train protocol."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, os.getcwd())
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.capri.sender import CapriSender
from opencood.models.capri.model import Capri
from opencood.models.capri.geometry import register_background
from opencood.tools.train_utils import to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(str(out))
    cfg = yaml.safe_load(Path(args.config).read_text())
    base = load_yaml(cfg['base_config'])
    base['ectra_ego_history'] = {'enabled': True, 'max_age_ms': 100}
    dataset = build_dataset(base, visualize=False, train=False)
    sender = CapriSender(base['model']['args'], cfg['sender']).cuda().eval()
    sender.load_state_dict(torch.load(cfg['single_checkpoint'], map_location='cpu'), strict=True)
    rows = []
    for index in range(0, min(len(dataset), 1601), 200):
        random.seed(303+index)
        np.random.seed(303+index)
        torch.manual_seed(303+index)
        raw = dataset[index]
        if raw is None:
            continue
        data = to_device(dataset.collate_batch_test([raw]), torch.device('cuda'))['ego']
        k = int(data['past_k_time_interval'].numel()//2)
        history = data['ectra_ego_history']
        sender.settings['background_points'] = 192
        messages = sender.messages(data['processed_lidar'], 2*k, data['anchor_box'], dataset.post_processor)
        refs = sender.messages(history['processed_lidar'], k, data['anchor_box'], dataset.post_processor)
        sender.settings['background_points'] = 2048
        dense = sender.messages(history['processed_lidar'], k, data['anchor_box'], dataset.post_processor)
        for frame in range(k):
            source = Capri.transform_message(messages[k+frame], data['pairwise_t_matrix'][0, 1, frame])
            row = {'index': index, 'frame': frame, 'history_valid': bool(history['valid'][0, frame])}
            if row['history_valid']:
                for name, reference in [('sparse', refs[frame]), ('dense_local', dense[frame])]:
                    target = Capri.transform_message(reference, history['to_current_ego'][0, frame])
                    transform, stats = register_background(source['background'], target['background'])
                    row[name] = dict(stats, source_points=len(source['background']['xyz']),
                                     target_points=len(target['background']['xyz']),
                                     correction=transform.cpu().tolist())
            rows.append(row)
            print(json.dumps(row), flush=True)
    out.write_text(json.dumps({'note': 'Stratified diagnostic, not AP or proof of true pose accuracy', 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
