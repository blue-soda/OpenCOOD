"""Bounded synthetic CUDA wiring audit; does not claim real-data accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from opencood.models.e3dsnn_single import E3dsnnSingle
from opencood.models.e3dsnn_single_ann import E3dsnnSingleAnn
from opencood.models.e3dsnn_fusion import E3dsnnFusion


def run():
    torch.manual_seed(20260930)
    torch.backends.cudnn.deterministic = True
    args = {'voxel_size': [.2, .2, .1], 'lidar_range': [-100.8, -40, -3.5, 100.8, 40, 1.5],
            'anchor_number': 2, 'dir_args': {'num_bins': 2}, 'message_dropout': 0.}
    rng = np.random.RandomState(20260930)
    coords = []
    for batch in range(2):
        q = np.unique(np.stack([rng.randint(0, 50, 5000), rng.randint(140, 260, 5000),
                                rng.randint(400, 600, 5000)], 1), axis=0)
        coords.append(np.column_stack([np.full(len(q), batch), q]))
    coords = np.concatenate(coords).astype(np.int32)
    xyz = (coords[:, [3, 2, 1]] + .5) * np.array([.2, .2, .1]) + np.array([-100.8, -40, -3.5])
    points = np.concatenate([xyz, np.ones((len(xyz), 1))], 1).astype(np.float32)
    lidar = {'voxel_coords': torch.from_numpy(coords).cuda(),
             'voxel_features': torch.from_numpy(np.repeat(points[:, None, :], 4, 1)).cuda(),
             'voxel_num_points': torch.full((len(coords),), 4, dtype=torch.int32, device='cuda')}
    transform = torch.eye(4, device='cuda').repeat(2, 1, 1)
    transform[:, 0, 3] = 1.6
    transform[:, 2, 3] = .3
    data = {'processed_lidar': lidar, 'road_lidar': lidar, 'object_bbx_mask': torch.ones(2, 1, device='cuda'),
            'road_to_ego': transform, 'frame_ids': ['synthetic0', 'synthetic1'],
            'timestamps_us': torch.tensor([[10, 20], [30, 40]], device='cuda')}
    report = {'synthetic_only': True, 'batch_size': 2, 'checks': {}}
    for activation, cls in [('count4', E3dsnnSingle), ('relu', E3dsnnSingleAnn)]:
        config = dict(args, activation=activation)
        single = cls(config).cuda().eval()
        fusion = E3dsnnFusion(config).cuda().eval()
        fusion.load_single(single.state_dict())
        with torch.no_grad():
            base = single(data)
            no_road = fusion(dict(data, disable_road=True))
            for key in ('psm', 'rm', 'dm'):
                torch.testing.assert_close(base[key], no_road[key], rtol=1e-5, atol=1e-5)
            # Hold encoder output fixed to isolate exact serialization from
            # the known independent sparse-CUDA forward nondeterminism.
            fusion.train()
            encoded = fusion.encode_lidar(lidar, 2)
            assert encoded.features.abs().sum() > 0, 'Packet audit must not be an all-zero message'
            fusion.eval()
            original_encoder = fusion.encode_lidar
            fusion.encode_lidar = lambda *_: encoded
            original_detector = fusion.detect_bev
            fused_inputs = []
            def capture(spatial):
                fused_inputs.append(spatial.detach().clone())
                return original_detector(spatial)
            fusion.detect_bev = capture
            direct = fusion(data)
            fusion.packet_roundtrip = True
            packed = fusion(data)
            for key in ('psm', 'rm', 'dm'):
                assert torch.equal(direct[key], packed[key]), (activation, key)
            assert torch.equal(fused_inputs[0], fused_inputs[1]) and fused_inputs[0].abs().sum() > 0
            fusion.packet_roundtrip = False
            fusion.fusion_mode = 'max'
            with_road = fusion(data)
            without_road = fusion(dict(data, disable_road=True))
            # Untrained Count4 BEV eval can yield zero heads; assert message
            # influence at the fusion output rather than inventing detections.
            assert (fused_inputs[2] - fused_inputs[3]).abs().max() > 1e-8
            fusion.detect_bev = original_detector
            fusion.encode_lidar = original_encoder
            fusion.packet_roundtrip = False
            fusion.fusion_mode = 'max'
            maximum = fusion(data)
            assert maximum['psm'].shape == (2, 2, 50, 126)
            assert all(torch.isfinite(maximum[k]).all() for k in ('psm', 'rm', 'dm'))
        del single, base, no_road, encoded, direct, maximum, with_road, without_road
        fusion.fusion_mode = 'residual'
        fusion.train()
        outputs = fusion(data)
        loss = sum(outputs[k].square().mean() for k in ('psm', 'rm', 'dm'))
        loss.backward()
        gradients = {}
        for name in ('backbone_3d.conv_input.0.weight', 'backbone_2d.blocks.0.2.weight',
                     'fusion_net.0.weight', 'fusion_net.3.weight', 'reg_head.weight'):
            grad = dict(fusion.named_parameters())[name].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
            gradients[name] = float(grad.abs().sum())
        report['checks'][activation] = {'no_road_matches_single': True, 'frozen_feature_packet_exact_heads': True,
            'packet_bytes': packed['packet_bytes'], 'max_forward_finite': True, 'max_uses_road_before_bev': True,
            'packet_exact_nonzero_fused_features': True,
            'packet_feature_source': 'synthetic batch, encoder BN in train mode; not a learned communication-size estimate',
            'gradient_l1': gradients, 'parameters': sum(p.numel() for p in fusion.parameters())}
        del fusion, outputs, loss, packed
        torch.cuda.empty_cache()
    assert report['checks']['count4']['parameters'] == report['checks']['relu']['parameters']
    report['passed'] = True
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run()
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
