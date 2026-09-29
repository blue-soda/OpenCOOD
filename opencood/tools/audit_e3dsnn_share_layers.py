"""Measure candidate sharing layers on fixed, evenly spaced validation frames."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import write_json
from opencood.utils.spike_packet import encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    random.seed(20260929)
    np.random.seed(20260929)
    torch.manual_seed(20260929)
    torch.cuda.manual_seed_all(20260929)
    config = 'opencood/hypes_yaml/dair-v2x/snn/e3dsnn_vehicle_single.yaml'
    cfg = load_yaml(config)
    dataset = build_dataset(cfg, train=False)
    model = train_utils.create_model(cfg).cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=int).tolist()
    write_json(folder / 'manifest.json', {'arguments': vars(args), 'checkpoint_sha256': digest,
        'config_sha256': hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        'data_manifest_sha256': hashlib.sha256(Path(cfg['data_manifest']).read_bytes()).hexdigest(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'indices': indices, 'frame_ids': [dataset.data[i] for i in indices]})
    captured = {}
    model.backbone_3d.conv_input[6].register_forward_pre_hook(lambda m, i: captured.update(early_spike=i[0]))
    for name in ('conv2', 'conv3', 'conv4', 'conv_out'):
        getattr(model.backbone_3d, name).register_forward_hook(
            lambda m, i, o, key=name: captured.update({key: o}))
    model.backbone_2d.register_forward_pre_hook(lambda m, i: captured.update(height_compressed=i[0]['spatial_features']))
    model.backbone_2d.register_forward_hook(lambda m, i, o: captured.update(bev_neck=o['spatial_features_2d']))
    rows = []
    for index in indices:
        batch = train_utils.to_device(dataset.collate_batch_test([dataset[index]]), torch.device('cuda'))
        captured.clear()
        with torch.no_grad():
            model(batch['ego'])
        result = {'frame_id': dataset.data[index], 'layers': {}}
        for name, tensor in captured.items():
            sparse = hasattr(tensor, 'features')
            if sparse:
                values = tensor.features.detach().cpu().numpy()
                coords = tensor.indices[:, 1:].detach().cpu().numpy()
                spatial = list(map(int, tensor.spatial_shape))
                coord_bytes = tensor.indices.numel() * tensor.indices.element_size()
            else:
                _, channels, height, width = tensor.shape
                spatial = [1, height, width]
                values = tensor[0].permute(1, 2, 0).reshape(-1, channels).detach().cpu().numpy()
                y, x = np.indices((height, width))
                coords = np.column_stack([np.zeros(height * width, dtype=np.int32), y.ravel(), x.ravel()])
                coord_bytes = 0
            is_count = bool(np.isfinite(values).all() and np.all(values == np.floor(values)) and np.all((values >= 0) & (values <= 4)))
            row = {'spatial_shape_zyx': spatial, 'rows': len(values), 'channels': values.shape[1],
                'count4': is_count, 'feature_bytes': values.nbytes, 'coordinate_storage_bytes': coord_bytes,
                'dense_fp32_bytes': int(np.prod(spatial)) * values.shape[1] * 4}
            if is_count:
                metadata = {'layer': 'backbone_3d.conv_out' if name == 'conv_out' else name,
                    'frame_id': dataset.data[index], 'model_sha256': digest, 'spatial_shape_zyx': spatial,
                    'coordinate_contract': 'sender-local sparse tensor indices; physical alignment pending'}
                row['packet_bytes'] = len(encode(coords, values, metadata))
                row['nonzero_rows'] = int(np.any(values != 0, axis=1).sum())
            result['layers'][name] = row
        rows.append(result)
        with (folder / 'frames.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        if len(rows) % 16 == 0:
            print(json.dumps({'completed': len(rows), 'samples': len(indices)}), flush=True)
    summary = {'samples': len(rows), 'checkpoint_sha256': digest, 'layers': {}}
    for name in rows[0]['layers']:
        entries = [row['layers'][name] for row in rows]
        summary['layers'][name] = {key: entries[0][key] for key in ('spatial_shape_zyx', 'channels', 'count4', 'dense_fp32_bytes')}
        for key in ('rows', 'feature_bytes', 'coordinate_storage_bytes', 'packet_bytes', 'nonzero_rows'):
            if key in entries[0]:
                summary['layers'][name][key + '_mean'] = float(np.mean([row[key] for row in entries]))
    write_json(folder / 'summary.json', summary)
    write_json(folder / 'status.json', {'state': 'complete'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
