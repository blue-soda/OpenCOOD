"""Stage-E0 sender-only IF stream adapter; frozen mature receiver/backend.

A separate experimental path. Not an already-trained detector, no event-kernel
efficiency claim. Shared unit threshold/horizon, sampled actual byte budgets,
no three-way partition and no fixed 25% spatial preselection.
"""
from dataclasses import asdict
import hashlib
import json
import numpy as np
import torch
from opencood.models.e3dsnn_joint_detector import E3dsnnJointDetector
from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire, stream_readout
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.utils import event_stream_packet as codec
from opencood.utils.e3dsnn_transplant import encoder_digest


class E3dsnnEventDetector(E3dsnnJointDetector):
    def __init__(self, args):
        super().__init__(args)
        if self.content_full_packet or self.joint_rate_weight or self.joint_prefix_weights:
            raise ValueError('Event path does not reuse three-prefix/rate objectives')
        options = args['event_stream']
        self.event_budget = options['budget_bytes']
        self.event_train_budgets = tuple(options['train_budget_range'])
        self.event_packet_events = options.get('packet_events', 256)
        self.event_max_spikes = options.get('max_spikes', 8)
        if (type(self.event_budget) is not int or self.event_budget < 0 or
                len(self.event_train_budgets) != 2 or
                any(type(x) is not int or x < 0 for x in self.event_train_budgets) or
                self.event_train_budgets[0] > self.event_train_budgets[1]):
            raise ValueError('Invalid event byte budgets')
        if not self.joint_receiver.retain_residual or not self.joint_receiver.use_local_membrane:
            raise ValueError('First sender experiment fixes full joint receiver')
        self.sender_ranking = 'analytical_IF_crossing_time_stable_address_ties'
        self.content_prefix_semantics = 'unit channel events truncated at serialized byte budget'
        self.joint_learn_order = False
        self.message_support_kind = 'cumulative_received_3d'
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith('content.project.'))

    def event_basis_digest(self):
        spec = dict(content=self.content.basis_digest(), threshold=1., horizon=1.,
                    max_spikes=self.event_max_spikes, schema='if-unit-events-v1')
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode('ascii')).hexdigest()

    def forward(self, data):
        batch_size = data['object_bbx_mask'].shape[0]
        ego_sparse, membrane = self.encode_membrane(data['processed_lidar'], batch_size)
        ego = ego_sparse.dense()
        if self.fusion_mode == 'ego' or data.get('disable_road', False):
            return dict(self.detect_bev(ego.flatten(1, 2)), message_kept=ego.new_zeros(batch_size),
                        packet_bytes=[0] * batch_size)
        # All occupied source features are candidates; no raw-L1 25% selector.
        road = self.encode_lidar(data['road_lidar'], batch_size).dense()
        if self._encoder_identity is None:
            self._encoder_identity = encoder_digest(self)
        basis = self.event_basis_digest()
        kept = ego.new_ones(batch_size)
        if self.training and self.message_dropout:
            kept = (torch.rand(batch_size, device=ego.device) >= self.message_dropout).to(ego.dtype)
        features, byte_counts, event_counts, budgets = [], [], [], []
        for i in range(batch_size):
            contract = MessageContract(str(data['frame_ids'][i]), str(data['road_frame_ids'][i]),
                'infrastructure', self._encoder_identity, 'event-content32-' + basis,
                tuple(int(n) for n in ego.shape[2:]), tuple(self.origin), tuple(self.step),
                tuple(int(n) for n in data['timestamps_us'][i].cpu().tolist()))
            pose = data['road_to_ego'][i:i+1]
            drive = torch.relu(self.content.project(road[i:i+1]))
            stream = integrate_and_fire(drive, self.event_max_spikes)
            budget = self.event_budget if not self.training else int(torch.randint(
                self.event_train_budgets[0], self.event_train_budgets[1]+1, (1,)).item())
            metadata = dict(contract=asdict(contract), source_to_ego=pose[0].cpu().tolist(),
                            channels=32, max_spikes=self.event_max_spikes, schema='if-unit-events-v1')
            packets, n, used = codec.pack_prefix(stream.addresses, metadata, budget,
                                                self.event_packet_events)
            counts = stream.prefix(n)
            if self.packet_roundtrip:
                if self.training:
                    raise ValueError('Wire reconstruction is an evaluation-only audit')
                receiver = codec.StreamAccumulator(metadata)
                for packet in packets:
                    receiver.receive(packet)
                decoded = counts.new_tensor(receiver.finish())
                if not torch.equal(decoded, counts):
                    raise ValueError('Direct/wire event mismatch')
                counts = decoded
            features.append(stream_readout(membrane[i:i+1], counts * kept[i], pose,
                                           contract, self.joint_receiver, self.content.decode))
            byte_counts.append(used); event_counts.append(n); budgets.append(budget)
        output = dict(self.detect_bev(torch.cat(features).flatten(1, 2)),
                      message_kept=kept.detach(), packet_bytes=byte_counts,
                      stream_events=event_counts, stream_budgets=budgets)
        if self.training:
            output['prefix_outputs'] = [{k: output[k] for k in ('psm', 'rm', 'dm')}]
            if self.enable_ego_aux:
                output['ego_output'] = self.detect_bev(ego.flatten(1, 2))
        return output
