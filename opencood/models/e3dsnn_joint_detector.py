"""Joint trainable Count3 content/order and frame-local membrane receiver.

Frozen mature SNN encoder and ANN BEV/continuous heads are the initial bounded
experiment. P/D, sender order and membrane gate are jointly trained. Learned
order assigns complete latent voxel rows to packets; it is not learned timing
of individual spikes and does not require receiver data at the sender.
"""
from dataclasses import asdict
import json
import math
import numpy as np
import torch
from opencood.models.e3dsnn_content_detector import E3dsnnContentDetector
from opencood.models.fuse_modules.e3dsnn_joint_membrane import JointMembraneReceiver
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.models.fuse_modules.e3dsnn_progressive_sender import ProgressiveSender, l1_partition
from opencood.utils import content_feature_packet as codec
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.utils.e3dsnn_rate_objective import sparse_payload_proxy


class E3dsnnJointDetector(E3dsnnContentDetector):
    def __init__(self, args):
        super().__init__(args)
        if self.activation != 'count4' or not self.content_trainable:
            raise ValueError('Joint prototype requires SNN encoder and learned Count3 content')
        self.joint_receiver = JointMembraneReceiver(
            retain_residual=bool(args.get('joint_retain_residual', True)),
            use_local_membrane=bool(args.get('joint_local_membrane', True)))
        self.progressive_sender = ProgressiveSender(channels=32)
        self.joint_learn_order = bool(args.get('joint_learn_order', True))
        self.joint_eval_order = args.get('joint_eval_order', 'learned')
        self.joint_prefix_weights = args.get('joint_prefix_weights')
        if self.joint_prefix_weights is not None:
            weights = self.joint_prefix_weights
            if (len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights)
                    or abs(sum(weights)-1.) > 1e-6):
                raise ValueError('Invalid joint prefix loss weights')
        self.joint_rate_weight = float(args.get('joint_rate_weight', 0.))
        if not math.isfinite(self.joint_rate_weight) or self.joint_rate_weight < 0:
            raise ValueError('Invalid joint rate objective weight')
        if self.joint_eval_order not in ('learned', 'l1'):
            raise ValueError('Unknown joint evaluation ordering')
        self.sender_ranking = 'learned_content32_voxel_batch' if self.joint_learn_order else 'frozen_content32_voxel_batch'
        if self.joint_eval_order == 'l1':
            self.sender_ranking = 'l1_content32_voxel_batch'
        self.message_support_kind = 'received_3d_before_membrane_update'
        self.content_prefix_semantics = 'learned disjoint voxel batches of Count3; content and order trained jointly'
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(('content.', 'joint_receiver.')) or
                             (self.joint_learn_order and name.startswith('progressive_sender.')))

    def load_content(self, state):
        missing, extra = self.load_state_dict(state, strict=False)
        expected = {k for k in self.state_dict() if k.startswith(('joint_receiver.', 'progressive_sender.'))}
        if set(missing) != expected or extra:
            raise ValueError('Unexpected content-to-joint checkpoint mapping')

    def forward(self, data):
        if self.training and self.joint_eval_order != 'learned':
            raise ValueError('Alternative joint ordering is an inference-only control')
        if self.training and (self.packet_roundtrip or self.readout_prefix != 3 or self.content_full_packet):
            raise ValueError('Training requires three differentiable prefixes')
        b = data['object_bbx_mask'].shape[0]
        ego_sparse, membrane = self.encode_membrane(data['processed_lidar'], b)
        ego = ego_sparse.dense()
        if ego.shape[1:3] != (128, 3):
            raise ValueError('Unexpected detector grid')
        ego_bev = ego.flatten(1, 2)
        if self.fusion_mode == 'ego' or data.get('disable_road', False):
            return dict(self.detect_bev(ego_bev), message_kept=ego.new_zeros(b))
        if len(data['frame_ids']) != b or len(data['road_frame_ids']) != b:
            raise ValueError('Per-sample identities required')
        road = self.select_road(self.encode_lidar(data['road_lidar'], b), b)
        if self._encoder_identity is None:
            self._encoder_identity = encoder_digest(self)
        basis = self.content.basis_digest()
        states, contracts, poses, chunks, rate_proxies = [], [], [], [], []
        lengths = [0] * b
        kept = ego.new_ones(b)
        if self.training and self.message_dropout:
            kept = (torch.rand(b, device=ego.device) >= self.message_dropout).to(ego.dtype)

        def wire(values, i, packet_index, transport):
            active = (values[0] != 0).any(0)
            coords = torch.nonzero(active).cpu().numpy()
            features = values[0, :, active].T.detach().cpu().numpy()
            meta = dict(contract=asdict(contracts[i]), basis_sha256=basis,
                source_to_ego=poses[i][0].cpu().tolist(), packet_index=packet_index,
                packet_count=3, schema='joint-content-voxel-batches-v1', transport=transport)
            scales = np.ones(32, dtype=np.float32)
            kind = 'joint_batch' if transport == 'progressive_voxel_batch' else 'joint_full'
            packet = codec.encode(coords, features, scales, meta, 'count3', kind)
            coords, features, decoded_scales, metadata = codec.decode(packet, 'count3', kind)
            if metadata != json.loads(json.dumps(meta)) or not np.array_equal(decoded_scales, scales):
                raise ValueError('Packet contract or scale changed')
            out = torch.zeros_like(values)
            pos = torch.as_tensor(coords.astype(np.int64), device=values.device)
            out[0, :, pos[:, 0], pos[:, 1], pos[:, 2]] = values.new_tensor(features.T.copy())
            lengths[i] += len(packet)
            return out

        for i in range(b):
            contract = MessageContract(str(data['frame_ids'][i]), str(data['road_frame_ids'][i]),
                'infrastructure', self._encoder_identity, 'joint-content32-' + basis,
                tuple(int(n) for n in ego.shape[2:]), tuple(self.origin), tuple(self.step),
                tuple(int(n) for n in data['timestamps_us'][i].cpu().tolist()))
            pose = data['road_to_ego'][i:i+1]
            contracts.append(contract); poses.append(pose)
            states.append(self.joint_receiver.begin(membrane[i:i+1], contract, pose,
                self._encoder_identity, basis))
            message = self.content.encode(road[i:i+1])
            counts = message.counts
            if self.training and self.joint_rate_weight:
                source_rows = int((road[i:i+1].detach() != 0).any(1).sum())
                rate_proxies.append(sparse_payload_proxy(counts, source_rows))
            if self.content_full_packet:
                if not self.packet_roundtrip:
                    raise ValueError('Full transport control requires wire roundtrip')
                # Receiver repeats the same sender-local scorer on received full
                # counts. This preserves the schedule for a transport-only test;
                # its extra computation is not an optimized single-pass latency.
                counts = wire(counts, i, 3, 'full_then_same_schedule')
            chunks.append(self.progressive_sender(counts, 3) if self.joint_eval_order == 'learned'
                          else l1_partition(counts, 3))
        outputs = []
        for k in range(self.readout_prefix):
            for i in range(b):
                part = chunks[i][k] * kept[i]
                if self.packet_roundtrip and not self.content_full_packet:
                    part = wire(part, i, k + 1, 'progressive_voxel_batch')
                self.joint_receiver.receive(states[i], part, contracts[i], poses[i], k + 1,
                                            self.content.decode)
            feature = torch.cat([s.output for s in states]).flatten(1, 2)
            outputs.append(self.detect_bev(feature))
        for state in states:
            self.joint_receiver.finish(state)
        result = dict(outputs[-1], message_kept=kept.detach())
        if self.training:
            result['prefix_outputs'] = outputs
            if self.joint_prefix_weights is not None:
                result['prefix_loss_weights'] = self.joint_prefix_weights
            if rate_proxies:
                result['message_rate_loss'] = self.joint_rate_weight * torch.stack(rate_proxies).mean()
            if self.enable_ego_aux:
                result['ego_output'] = self.detect_bev(ego_bev)
        if self.packet_roundtrip:
            result['packet_bytes'] = lengths
        return result
