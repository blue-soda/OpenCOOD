"""Sender 3D messages -> SE(3) warp -> height fold -> matched BEV fusion."""
import torch
from torch import nn
from opencood.models.e3dsnn_single import E3dsnnSingle
from opencood.models.e3dsnn_single_ann import replace_counts
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
from opencood.models.fuse_modules.e3dsnn_geometry import encoded_grid, warp_volume
from opencood.utils import fusion_packet


class E3dsnnFusion(E3dsnnSingle):
    def __init__(self, args):
        super().__init__(args)
        self.activation = args.get('activation', 'count4')
        self.fusion_mode = args.get('fusion_mode', 'residual')
        if self.fusion_mode not in ('ego', 'max', 'residual'):
            raise ValueError('Unknown fusion mode')
        self.origin, self.step = encoded_grid(args['voxel_size'], args['lidar_range'])
        self.packet_roundtrip = False
        self.message_dropout = args.get('message_dropout', .1)
        depth = 3
        # Current checkpoint contract is C=128,D=3,H=50,W=126.
        channels = 128 * depth
        self.fusion_net = nn.Sequential(nn.Conv2d(2 * channels + depth, 64, 1, bias=False),
            nn.GroupNorm(8, 64), Multispike(), nn.Conv2d(64, channels, 1))
        nn.init.normal_(self.fusion_net[-1].weight, std=.01)
        nn.init.zeros_(self.fusion_net[-1].bias)
        self.fusion_scale = nn.Parameter(torch.tensor(.1))
        self.fusion_activation = Multispike()
        if self.activation == 'relu':
            replace_counts(self)
        elif self.activation != 'count4':
            raise ValueError('Unknown activation')

    def load_single(self, state):
        missing, extra = self.load_state_dict(state, strict=False)
        expected = {n for n in self.state_dict() if n.startswith('fusion_net.') or n == 'fusion_scale'}
        if set(missing) != expected or extra:
            raise ValueError('Unexpected pretrained mapping: {}, {}'.format(missing, extra))

    def forward(self, data):
        b = data['object_bbx_mask'].shape[0]
        ego_sparse = self.encode_lidar(data['processed_lidar'], b)
        ego = ego_sparse.dense()
        if tuple(ego.shape[1:3]) != (128, 3):
            raise ValueError('Fusion requires the audited 128-channel, 3-height grid')
        ego_bev = ego.flatten(1, 2)
        if self.fusion_mode == 'ego' or data.get('disable_road', False):
            result = self.detect_bev(ego_bev)
            if data.get('return_spikes', False):
                result['spike_features'] = ego_sparse.features
            return result
        road_sparse = self.encode_lidar(data['road_lidar'], b)
        transform = data['road_to_ego']
        packet_bytes = []
        if self.packet_roundtrip:
            if self.training:
                raise ValueError('Serialization is eval-only; train on differentiable sender features')
            road = torch.zeros_like(ego)
            matrices = []
            for i in range(b):
                active = road_sparse.indices[:, 0] == i
                meta = {'schema': 1, 'sender': 'infrastructure', 'frame_id': data['frame_ids'][i],
                    'timestamps_us': data['timestamps_us'][i].detach().cpu().tolist(),
                    'shape_zyx': list(ego.shape[2:]), 'origin_xyz': self.origin, 'step_xyz': self.step,
                    'source_to_ego': transform[i].detach().cpu().tolist()}
                packet = fusion_packet.encode(road_sparse.indices[active, 1:].cpu().numpy(),
                    road_sparse.features[active].detach().cpu().numpy(), meta, self.activation)
                coords, values, received = fusion_packet.decode(packet, self.activation)
                if received != meta:
                    raise ValueError('Packet metadata changed')
                xyz = torch.as_tensor(coords.astype('int64'), device=ego.device)
                road[i, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = torch.as_tensor(values.T.copy(), device=ego.device, dtype=ego.dtype)
                matrices.append(ego.new_tensor(received['source_to_ego']))
                packet_bytes.append(len(packet))
            transform = torch.stack(matrices)
        else:
            road = road_sparse.dense()
        aligned, coverage = warp_volume(road, transform, self.origin, self.step)
        road_bev, mask = aligned.flatten(1, 2), coverage.flatten(1, 2)
        if self.training and self.message_dropout:
            keep = (torch.rand(b, 1, 1, 1, device=ego.device) >= self.message_dropout).to(ego.dtype)
            road_bev, mask = road_bev * keep, mask * keep
        supported = mask.amax(1, keepdim=True)
        if self.fusion_mode == 'max':
            fused = self.fusion_activation(torch.maximum(ego_bev, road_bev))
        else:
            delta = self.fusion_net(torch.cat((ego_bev, road_bev, mask), 1))
            fused = self.fusion_activation(ego_bev + self.fusion_scale * supported * delta)
        fused = torch.where(supported > 0, fused, ego_bev)
        result = self.detect_bev(fused)
        if data.get('return_spikes', False):
            result['spike_features'] = ego_sparse.features
        if packet_bytes:
            result['packet_bytes'] = packet_bytes
        return result
