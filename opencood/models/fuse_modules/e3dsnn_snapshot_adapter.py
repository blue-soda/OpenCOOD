"""Shared actual-byte P/D adaptation for two historical frozen detectors."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire, stream_readout
from opencood.models.fuse_modules.e3dsnn_content_message import CountContent
from opencood.models.fuse_modules.e3dsnn_snapshot_selection import select_training_snapshot, select_event_quota, select_if_window
from opencood.models.fuse_modules.e3dsnn_snapshot_readout import ann_readout
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.utils import task_snapshot_packet as wire

TRAINABLE = ('content.project.weight', 'content.expand.weight')


def shuffle_positive_counts(counts, frame_id):
    """Permute multiplicities within each channel, preserving its active voxels."""
    if counts.ndim != 5 or counts.shape[0] != 1:
        raise ValueError('Expected one decoded [1,C,Z,Y,X] count message')
    seed = int.from_bytes(hashlib.sha256(str(frame_id).encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    shuffled = counts.copy(order='C')
    for channel in shuffled[0].reshape(shuffled.shape[1], -1):
        active = np.flatnonzero(channel)
        channel[active] = rng.permutation(channel[active])
    return shuffled


class SnapshotAdapter:
    def __init__(self, args):
        super().__init__(args)
        self.snapshot_budget = args['snapshot']['budget_bytes']
        self.snapshot_family = args['snapshot']['family']
        self.shuffle_received_counts = False
        self.snapshot_selection = args['snapshot'].get('selection', 'actual_bytes')
        self.snapshot_order = args['snapshot'].get('order', 'time')
        if self.snapshot_order not in ('time', 'block', 'random'):
            raise ValueError('Unknown snapshot event order')
        self.snapshot_event_quotas = {int(k):v for k,v in args['snapshot'].get('event_quotas', {}).items()}
        self.snapshot_split_links = int(args['snapshot'].get('split_links', 1))
        self.snapshot_split_mode = args['snapshot'].get('split_mode', 'flat')
        self.snapshot_codec = args['snapshot'].get('codec', 'v1')
        self.snapshot_drop_links = int(args['snapshot'].get('drop_links', 0))
        if not 0 <= self.snapshot_drop_links < 4:
            raise ValueError('drop_links must be in [0, 4)')
        if self.snapshot_codec not in ('v1', 'v2', 'v3'):
            raise ValueError('Unknown snapshot codec version')
        if self.snapshot_split_mode not in ('flat', 'channel'):
            raise ValueError('Unknown message split mode')
        if not 1 <= self.snapshot_split_links <= 4:
            raise ValueError('Split links must be between 1 and 4')
        self.snapshot_measure_bytes = False
        if self.snapshot_selection not in ('actual_bytes', 'event_quota', 'if_window'):
            raise ValueError('Unknown snapshot selection policy')
        if self.snapshot_selection == 'if_window':
            self.if_window = float(args['snapshot']['if_window'])
            if self.snapshot_family != 'if8' or not 0 <= self.if_window <= 1 or self.snapshot_budget is not None:
                raise ValueError('Fixed-time selection requires IF8, a window in [0,1], and no byte budget')
        if self.snapshot_selection == 'event_quota':
            if self.snapshot_family != 'if8' or not self.snapshot_event_quotas:
                raise ValueError('Event quotas require IF8 and an explicit nominal-budget mapping')
            if any(type(v) is not int or v < 0 for v in self.snapshot_event_quotas.values()):
                raise ValueError('Event quotas must be nonnegative integers')
        if self.snapshot_family not in ('if8', 'ann2bit'):
            raise ValueError('Unknown snapshot family')
        if self.snapshot_budget is not None and (type(self.snapshot_budget) is not int or self.snapshot_budget < 0):
            raise ValueError('Invalid actual byte budget')
        self.snapshot_audit = False
        self.snapshot_records = []
        self.snapshot_packet_directory = args['snapshot'].get('packet_directory')
        self.snapshot_packet_index = 0
        # Same fused detection objective; the frozen ego auxiliary has no P/D
        # gradient and is disabled in both families to avoid redundant work.
        self.enable_ego_aux = False;self.ego_aux_weight = 0.
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in TRAINABLE)
        assert {n for n,p in self.named_parameters() if p.requires_grad} == set(TRAINABLE)
        self.sender_ranking = ('IF crossing time' if self.snapshot_family=='if8' else 'original raw L1 rows')
        self.content_prefix_semantics = 'single actual-byte snapshot; fixed selection surrogate'
        if self.snapshot_selection == 'event_quota':
            self.content_prefix_semantics = 'fixed IF event quota; bytes measured only on explicit export, no hard compressed-byte cap'
        elif self.snapshot_selection == 'if_window':
            self.sender_ranking = 'local IF threshold crossings before a fixed time'
            self.content_prefix_semantics = 'fixed IF window; event count and bytes vary by frame'

    def train(self, mode=True):
        super().train(mode)
        # No BN statistics, dropout layers or frozen fusion state may drift.
        for module in self.children():module.eval()
        self.content.train(mode)
        return self

    def snapshot_drive(self, source):
        # Default F0 arithmetic is unchanged; candidate subclasses may modulate
        # the source-only IF drive and must extend event_basis_digest as well.
        return torch.relu(self.content.project(source))

    def detect_bev_sample(self, spatial, sample_index):
        """Per-example dispatch for receivers with local multiscale features."""
        return self.detect_bev(spatial)

    def forward(self, data):
        if self.training and self.packet_roundtrip:
            raise ValueError('Wire reconstruction is eval-only; preserve training gradients')
        fixed_window = self.snapshot_selection == 'if_window'
        if fixed_window and self.training:
            raise ValueError('Fixed-window IF selection is an evaluation intervention')
        quota_mode = self.snapshot_selection == 'event_quota'
        tensor_selection = quota_mode or fixed_window
        observer = getattr(self, '_snapshot_packet_observer', None)
        measure = tensor_selection and not self.training and bool(self.snapshot_measure_bytes or
            self.packet_roundtrip or observer is not None or self.snapshot_packet_directory)
        if quota_mode and self.training and (self.snapshot_measure_bytes or observer is not None or self.snapshot_packet_directory):
            raise ValueError('Training must not encode, decode, export, or measure communication')
        b = data['object_bbx_mask'].shape[0]
        sparse, membrane = self.encode_membrane(data['processed_lidar'], b)
        ego = sparse.dense()
        local_only = self.fusion_mode=='ego' or data.get('disable_road', False)
        kept = ego.new_ones(b)
        if self.training and self.message_dropout:
            kept = (torch.rand(b,device=ego.device)>=self.message_dropout).to(ego.dtype)
        if not local_only:
            road = self.encode_lidar(data['road_lidar'], b)
            if self.snapshot_family=='if8':sources=road.dense()
            else:sources=self.select_road(road,b)
            if self._encoder_identity is None:self._encoder_identity=encoder_digest(self)
            basis=self.event_basis_digest() if self.snapshot_family=='if8' else self.content.basis_digest()
        outputs=[];sizes=[];units=[];self.snapshot_records=[]
        for i in range(b):
            if local_only:
                outputs.append(self.detect_bev_sample(ego[i:i+1].flatten(1,2),i));sizes.append(0);units.append(0)
                continue
            if fixed_window:
                full = self.snapshot_drive(sources[i:i+1])
            elif self.snapshot_family=='if8':
                stream=integrate_and_fire(self.snapshot_drive(sources[i:i+1]),8,order_mode=self.snapshot_order)
                full=stream.drive if quota_mode else stream.prefix(len(stream.addresses))
                scales=full.new_ones((1,32,1,1,1))
                order=dict(stream=stream)
            else:
                message=self.content.encode(sources[i:i+1]);full,scales=message.counts,message.scales
                pos=road.indices[road.indices[:,0]==i,1:].cpu().numpy();shape=full.shape[2:]
                flat=(pos[:,0]*shape[1]+pos[:,1])*shape[2]+pos[:,2]
                score=road.features[road.indices[:,0]==i].detach().cpu().numpy().sum(1,dtype=np.float64)
                rank=np.lexsort((flat,-score))
                active=np.flatnonzero(np.any(full.detach().cpu().numpy().reshape(32,-1)!=0,axis=0))
                order=dict(row_order=flat[rank][np.isin(flat[rank],active)])
            contract=MessageContract(str(data['frame_ids'][i]),str(data['road_frame_ids'][i]),
                'infrastructure',self._encoder_identity,'task32-'+basis,
                tuple(int(x) for x in full.shape[2:]),tuple(self.origin),tuple(self.step),
                tuple(int(x) for x in data['timestamps_us'][i].cpu().tolist()))
            pose=data['road_to_ego'][i:i+1]
            meta=dict(contract=asdict(contract),source_to_ego=pose[0].cpu().tolist(),
                family=self.snapshot_family,schema='task-count32-v1')
            if tensor_selection:
                if fixed_window:
                    selected = select_if_window(full, self.if_window)
                else:
                    if self.snapshot_budget in (0,None): quota=self.snapshot_budget
                    else:
                        if self.snapshot_budget not in self.snapshot_event_quotas:
                            raise ValueError('Unregistered nominal budget for event-quota selection')
                        quota=self.snapshot_event_quotas[self.snapshot_budget]
                    selected=select_event_quota(stream,quota)
                if measure:
                    if self.snapshot_split_links > 1:
                        if not self.packet_roundtrip:
                            raise ValueError('Split links require wire reconstruction')
                        flat=selected.counts.reshape(-1)
                        positions=torch.arange(flat.numel(),device=flat.device)
                        parts=[];total=0
                        for link in range(self.snapshot_split_links):
                            if self.snapshot_split_mode == 'channel':
                                channels=selected.counts.shape[1]
                                index=torch.arange(channels,device=flat.device).repeat_interleave(flat.numel()//channels)
                                mask=(index % self.snapshot_split_links == link)
                            else:
                                mask=(positions % self.snapshot_split_links == link)
                            part=(flat*mask).reshape(selected.counts.shape)
                            packet=wire.encode(part.detach().cpu().numpy(),
                                selected.scales.detach().cpu().numpy().reshape(32),meta)
                            parts.append(packet);total+=len(packet)
                        selected.parts=parts;selected.packet=parts[0]
                        selected.record.update(communication_measured=True,actual_bytes=total,
                            split_links=self.snapshot_split_links)
                    else:
                        encoder = {'v1': wire.encode, 'v2': wire.encode_v2, 'v3': wire.encode_v3}[self.snapshot_codec]
                        selected.packet=encoder(selected.counts.detach().cpu().numpy(),
                            selected.scales.detach().cpu().numpy().reshape(32),meta)
                        selected.record.update(communication_measured=True,actual_bytes=len(selected.packet))
            else:
                selected=select_training_snapshot(full,scales,meta,self.snapshot_budget,**order)
            counts=selected.counts;scales=selected.scales
            if self.packet_roundtrip:
                if self.snapshot_split_links > 1:
                    merged=None
                    surviving = selected.parts[self.snapshot_drop_links:] if self.snapshot_drop_links else selected.parts
                    if not surviving:
                        raise ValueError('All packets dropped; use a smaller drop_links')
                    for packet in surviving:
                        decoded,paid=wire.decode(packet,meta)
                        piece=full.new_tensor(decoded)
                        merged=piece if merged is None else merged+piece
                    counts=merged;scales=full.new_tensor(paid).reshape(1,32,1,1,1)
                else:
                    decoder = {'v1': wire.decode, 'v2': wire.decode_v2, 'v3': wire.decode_v3}[self.snapshot_codec]
                    decoded,paid=decoder(selected.packet,meta)
                    if self.shuffle_received_counts:
                        decoded = shuffle_positive_counts(decoded, contract.receiver_frame)
                    counts=full.new_tensor(decoded);scales=full.new_tensor(paid).reshape(1,32,1,1,1)
            counts=counts*kept[i]
            if self.snapshot_family=='if8':
                fused=stream_readout(membrane[i:i+1],counts,pose,contract,self.joint_receiver,self.content.decode).flatten(1,2)
            else:
                events=None if self.packet_roundtrip else [CountContent(counts,scales).event(k) for k in (1,2,3)]
                _,fused=ann_readout(self,ego[i:i+1],counts,scales,contract,pose,events)
            outputs.append(self.detect_bev_sample(fused,i))
            if selected.packet is not None:sizes.append(int(selected.record.get('actual_bytes', len(selected.packet))))
            units.append(selected.record['selected_events'])
            if observer is not None:
                if self.training or not self.packet_roundtrip:
                    raise ValueError('Packet observer requires real wire evaluation')
                observer(contract.receiver_frame, selected.packet)
            if self.snapshot_audit:
                self.snapshot_records.append(dict(selected.record,packet_sha256=hashlib.sha256(selected.packet).hexdigest() if selected.packet is not None else None,
                    frame_id=contract.receiver_frame,basis=basis))
            if self.snapshot_packet_directory:
                if self.training or not self.packet_roundtrip:
                    raise ValueError('Packet archive requires real wire evaluation')
                if self.snapshot_split_links > 1:
                    raise ValueError('Split-link runs do not archive a single packet per frame')
                directory=Path(self.snapshot_packet_directory)
                if self.snapshot_packet_index==0:directory.mkdir(parents=True,exist_ok=False)
                filename='{:06d}.bin'.format(self.snapshot_packet_index)
                (directory/filename).write_bytes(selected.packet)
                record=dict(selected.record,frame_id=contract.receiver_frame,file=filename,
                    sha256=hashlib.sha256(selected.packet).hexdigest(),basis=basis)
                with (directory/'index.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                self.snapshot_packet_index+=1
        keys=('psm','rm','dm')+(('qm',) if all('qm' in o for o in outputs) else ())
        output={k:torch.cat([o[k] for o in outputs]) for k in keys}
        output.update(message_kept=kept.detach(),stream_events=units)
        if tensor_selection:
            if quota_mode:
                output['event_quotas']=[self.snapshot_event_quotas.get(self.snapshot_budget,self.snapshot_budget)]*b
            output['communication_measured']=measure
            if measure:output['packet_bytes']=sizes
        else:
            output.update(packet_bytes=sizes,
                stream_budgets=[self.snapshot_budget if self.snapshot_budget is not None else n for n in sizes])
        return output
