"""Learned Count3/ANN2bit message content with mature residual receivers."""
import math
from dataclasses import asdict
import numpy as np
import torch
from opencood.models.e3dsnn_budget import E3dsnnBudget
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
from opencood.models.fuse_modules.e3dsnn_content_message import CountContentBottleneck
from opencood.models.fuse_modules.e3dsnn_content_receiver import ContentAccumulator
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.models.fuse_modules.e3dsnn_readout import residual_readout
from opencood.utils import content_feature_packet as codec
from opencood.utils.e3dsnn_transplant import encoder_digest


class E3dsnnContentDetector(E3dsnnBudget):
    def __init__(self, args):
        if args['train_encoder'] or args.get('fusion_mode', 'residual') not in ('residual', 'ego'):
            raise ValueError('Content protocol freezes encoder and uses residual/ego mode')
        super().__init__(args)
        self.content_family = 'count3' if self.activation == 'count4' else 'ann2bit'
        self.fusion_activation_kind = args['fusion_activation_kind']
        if self.fusion_activation_kind != self.activation:
            raise ValueError('Registered family control matches encoder and fusion activation')
        if self.activation == 'count4':
            self.fusion_net[2] = Multispike(); self.fusion_activation = Multispike()
        self.content = CountContentBottleneck(family=self.content_family)
        self.content_trainable = bool(args['content_trainable'])
        self.readout_prefix = int(args.get('readout_prefix', 3))
        self.content_full_packet = bool(args.get('content_full_packet', False))
        if not 1 <= self.readout_prefix <= 3: raise ValueError('Invalid prefix')
        if self.content_full_packet and self.readout_prefix != 3: raise ValueError('Full packet needs full prefix')
        self.sender_ranking = 'original_features_l1_then_content'
        self.message_support_kind = 'geometry'
        self._encoder_identity = None
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith('fusion_net.') or name == 'fusion_scale' or
                             (self.content_trainable and name.startswith('content.')))

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs); self._encoder_identity = None
        return result

    def load_legacy(self, state):
        missing, extra = self.load_state_dict(state, strict=False)
        expected = {k for k in self.state_dict() if k.startswith('content.')}
        if set(missing) != expected or extra: raise ValueError('Unexpected content transplant')

    def encode_membrane(self, lidar, batch_size):
        captured = {}
        def before(module, inputs): captured['value'] = inputs[0]
        handle = self.backbone_3d.conv_out[-1].register_forward_pre_hook(before)
        try:
            sparse = self.encode_lidar(lidar, batch_size)
            membrane = sparse.replace_feature(captured['value']).dense()
        finally: handle.remove()
        return sparse, membrane

    def select_road(self, sparse, batch_size):
        # Frozen raw encoder features determine locations, before learned content.
        xyz = sparse.indices.detach().cpu().numpy()
        values = sparse.features.detach().cpu().numpy()
        dense = sparse.features.new_zeros((batch_size, 128) + tuple(sparse.spatial_shape))
        for i in range(batch_size):
            rows = np.flatnonzero((xyz[:, 0] == i) & np.any(values != 0, axis=1))
            c = xyz[rows, 1:]; score = values[rows].sum(1, dtype=np.float64)
            order = np.lexsort((c[:, 2], c[:, 1], c[:, 0], -score))
            rows = rows[order[:int(math.ceil(len(rows) * self.budget_codec['keep_fraction']))]]
            index = torch.as_tensor(rows, device=dense.device)
            pos = sparse.indices[index, 1:].long()
            dense[i, :, pos[:, 0], pos[:, 1], pos[:, 2]] = sparse.features[index].T
        return dense

    def forward(self, data):
        if self.training and (self.packet_roundtrip or self.readout_prefix != 3 or self.content_full_packet):
            raise ValueError('Training requires three differentiable prefixes')
        b = data['object_bbx_mask'].shape[0]
        ego_sparse, local_membrane = self.encode_membrane(data['processed_lidar'], b)
        ego = ego_sparse.dense(); ego_bev = ego.flatten(1, 2)
        if ego.shape[1:3] != (128, 3): raise ValueError('Unexpected detector grid')
        if self.fusion_mode == 'ego' or data.get('disable_road', False):
            return dict(self.detect_bev(ego_bev), message_kept=ego.new_zeros(b))
        if len(data['frame_ids']) != b or len(data['road_frame_ids']) != b:
            raise ValueError('Per-sample identities required')
        road = self.select_road(self.encode_lidar(data['road_lidar'], b), b)
        if self._encoder_identity is None: self._encoder_identity = encoder_digest(self)
        basis = self.content.basis_digest(); receiver = ContentAccumulator(basis, self.content_family)
        messages, states, contracts, poses = [], [], [], []
        kept = ego.new_ones(b)
        if self.training and self.message_dropout:
            kept = (torch.rand(b, device=ego.device) >= self.message_dropout).to(ego.dtype)
        for i in range(b):
            message = self.content.encode(road[i:i+1]); messages.append(message)
            contract = MessageContract(str(data['frame_ids'][i]), str(data['road_frame_ids'][i]),
                'infrastructure', self._encoder_identity, 'content32-' + basis,
                tuple(int(n) for n in ego.shape[2:]), tuple(self.origin), tuple(self.step),
                tuple(int(n) for n in data['timestamps_us'][i].cpu().tolist()))
            pose = data['road_to_ego'][i:i+1]; poses.append(pose); contracts.append(contract)
            states.append(receiver.begin(message.counts, contract, pose, message.scales))
        packet_lengths = [0] * b
        # Optional single 2-bit full packet is an eval-only transport control.
        full_counts = []
        def wire(values, i, index, kind):
            active = (values[0] != 0).any(0)
            pos = torch.nonzero(active).cpu().numpy()
            codes = values[0, :, active].T.detach().cpu().numpy()
            meta = dict(contract=asdict(contracts[i]), basis_sha256=basis,
                source_to_ego=poses[i][0].cpu().tolist(), packet_index=index, packet_count=3)
            packet = codec.encode(pos, codes, messages[i].scales.flatten().detach().cpu().numpy(), meta, self.content_family, kind)
            pos, codes, scales, received = codec.decode(packet, self.content_family, kind)
            # JSON converts tuples to lists; compare canonical serialized data.
            import json
            if received != json.loads(json.dumps(meta)): raise ValueError('Content metadata changed')
            reconstructed = torch.zeros_like(values)
            index_tensor = torch.as_tensor(pos.astype(np.int64), device=ego.device)
            reconstructed[0, :, index_tensor[:, 0], index_tensor[:, 1], index_tensor[:, 2]] = values.new_tensor(codes.T.copy())
            if not torch.equal(values.new_tensor(scales).reshape(1,32,1,1,1), messages[i].scales):
                raise ValueError('Content frame scales changed')
            packet_lengths[i] += len(packet)
            return reconstructed
        if self.content_full_packet:
            if not self.packet_roundtrip: raise ValueError('Full packet control needs wire roundtrip')
            full_counts = [wire(m.counts, i, 3, 'full') for i, m in enumerate(messages)]
        outputs = []
        for k in range(1, self.readout_prefix + 1):
            aligned, masks = [], []
            for i, message in enumerate(messages):
                event = message.event(k) * kept[i]
                if self.content_full_packet: event = (full_counts[i] >= k).to(ego.dtype)
                elif self.packet_roundtrip: event = wire(event, i, k, 'event')
                latent, coverage = receiver.receive(states[i], event, contracts[i], poses[i], message.scales, k)
                aligned.append(self.content.decode(latent))
                nonempty = (states[i].source_counts.detach().abs().sum() > 0).to(ego.dtype)
                masks.append(coverage * kept[i] * nonempty)
            road_bev = torch.cat(aligned).flatten(1, 2); mask = torch.cat(masks).flatten(1, 2)
            supported = mask.amax(1, keepdim=True)
            fusion_input = torch.cat((ego_bev, road_bev, mask), 1)
            # Count thresholds can amplify batch-size-dependent CUDA convolution
            # rounding. Use the same single-frame fusion kernel for train/eval;
            # the real augmented batch audit verifies isolation of both families.
            delta = torch.cat([self.fusion_net(x[None]) for x in fusion_input])
            residual = self.fusion_scale * supported * delta
            fused = residual_readout(ego_bev, residual, supported, self.fusion_activation, self.fusion_readout)
            fused = torch.where(supported > 0, fused, ego_bev)
            outputs.append(self.detect_bev(fused))
        for state in states: receiver.finish(state)
        # local_membrane is held for this frame only; local features are never
        # replaced by an earlier fused prefix or another vehicle's membrane.
        del local_membrane
        result = dict(outputs[-1], message_kept=kept.detach())
        if self.training:
            result['prefix_outputs'] = outputs
            if self.enable_ego_aux: result['ego_output'] = self.detect_bev(ego_bev)
        if self.packet_roundtrip: result['packet_bytes'] = packet_lengths
        return result
