"""Sender 3D messages -> SE(3) warp -> height fold -> matched BEV fusion."""
import torch
from torch import nn
from opencood.models.e3dsnn_single import E3dsnnSingle
from opencood.models.e3dsnn_single_ann import replace_counts
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
from opencood.models.fuse_modules.e3dsnn_geometry import encoded_grid, warp_volume
from opencood.models.fuse_modules.e3dsnn_readout import residual_readout
from opencood.utils import fusion_packet


class E3dsnnFusion(E3dsnnSingle):
    def __init__(self, args):
        super().__init__(args)
        self.activation = args.get('activation', 'count4')
        self.fusion_mode = args.get('fusion_mode', 'residual')
        if self.fusion_mode not in ('ego', 'max', 'residual'):
            raise ValueError('Unknown fusion mode')
        self.fusion_readout = args.get('fusion_readout', 'activated')
        if self.fusion_readout not in ('activated', 'continuous'):
            raise ValueError('Unknown fusion readout')
        if self.fusion_readout == 'continuous' and self.fusion_mode != 'residual':
            raise ValueError('Continuous readout requires residual fusion')
        self.origin, self.step = encoded_grid(args['voxel_size'], args['lidar_range'])
        self.packet_roundtrip = False
        self.message_codec = args.get('message_codec')
        if self.message_codec is not None:
            from opencood.utils.quantized_feature_packet import validate_options
            validate_options(self.message_codec)
        self.collect_fusion_diagnostics = False
        self.freeze_detector = args.get('freeze_detector', False)
        self.freeze_detector_bn = args.get('freeze_detector_bn', False)
        self.enable_ego_aux = args.get('enable_ego_aux', False)
        self.ego_aux_weight = float(args.get('ego_aux_weight', 0.))
        if self.ego_aux_weight < 0 or (self.enable_ego_aux and not self.freeze_detector_bn):
            raise ValueError('Auxiliary controls require nonnegative weight and fixed BN statistics')
        self.message_dropout = args.get('message_dropout', .1)
        depth = 3
        # Current checkpoint contract is C=128,D=3,H=50,W=126.
        channels = 128 * depth
        self.fusion_net = nn.Sequential(nn.Conv2d(2 * channels + depth, 64, 1, bias=False),
            nn.GroupNorm(8, 64), Multispike(), nn.Conv2d(64, channels, 1))
        nn.init.normal_(self.fusion_net[-1].weight, std=.01)
        nn.init.zeros_(self.fusion_net[-1].bias)
        if args.get('identity_fusion_init', False):
            nn.init.zeros_(self.fusion_net[-1].weight)
        self.fusion_scale = nn.Parameter(torch.tensor(.1))
        self.fusion_activation = Multispike()
        if self.activation == 'relu':
            replace_counts(self)
        elif self.activation != 'count4':
            raise ValueError('Unknown activation')
        if self.freeze_detector:
            if self.fusion_mode != 'residual':
                raise ValueError('Frozen detector requires residual fusion')
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith('fusion_net.') or name == 'fusion_scale')

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_detector:
            # Freeze running statistics as well as weights, on every train() call.
            for name, module in self.named_children():
                if name not in ('fusion_net', 'fusion_activation'):
                    module.eval()
        elif self.freeze_detector_bn:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def load_single(self, state):
        missing, extra = self.load_state_dict(state, strict=False)
        expected = {n for n in self.state_dict() if n.startswith('fusion_net.') or n == 'fusion_scale'}
        if set(missing) != expected or extra:
            raise ValueError('Unexpected pretrained mapping: {}, {}'.format(missing, extra))

    def process_road_features(self, sparse):
        return sparse

    def wire_options(self):
        return self.message_codec

    def forward(self, data):
        if self.message_codec is not None and (self.training or not self.packet_roundtrip):
            raise ValueError('Rate probes require eval with actual packet roundtrip')
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
        road_sparse = self.process_road_features(road_sparse)
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
                codec, codec_args = fusion_packet, ()
                if self.wire_options() is not None:
                    from opencood.utils import quantized_feature_packet
                    codec, codec_args = quantized_feature_packet, (self.wire_options(),)
                packet = codec.encode(road_sparse.indices[active, 1:].cpu().numpy(),
                    road_sparse.features[active].detach().cpu().numpy(), meta, self.activation, *codec_args)
                coords, values, received = codec.decode(packet, self.activation, *codec_args)
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
            residual = self.fusion_scale * supported * delta
            fused = residual_readout(ego_bev, residual, supported,
                                     self.fusion_activation, self.fusion_readout)
        fused = torch.where(supported > 0, fused, ego_bev)
        result = self.detect_bev(fused)
        if self.training and self.enable_ego_aux:
            # Both lambda=0 and lambda=1 controls execute the same ego branch.
            # Fixed BN buffers prevent an extra pass from changing running stats.
            result['ego_output'] = self.detect_bev(ego_bev)
        if self.collect_fusion_diagnostics:
            if self.fusion_mode != 'residual' or self.training:
                raise ValueError('Fusion diagnostics require residual eval')
            from opencood.utils.e3dsnn_fusion_probe import probe_maps
            result['fusion_probe'] = probe_maps(ego_bev, residual, fused, supported,
                getattr(self, 'fusion_activation_kind', getattr(self, 'receiver_activation', self.activation))
                if self.fusion_readout == 'activated' else 'continuous')
        if data.get('return_spikes', False):
            result['spike_features'] = ego_sparse.features
        if packet_bytes:
            result['packet_bytes'] = packet_bytes
        return result
