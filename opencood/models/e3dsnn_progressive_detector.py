"""Frame-local membrane fusion with frozen, learned or spatial sender ordering."""
from dataclasses import asdict
import numpy as np
import torch
from opencood.models.e3dsnn_budget import E3dsnnBudget
from opencood.models.fuse_modules.e3dsnn_progressive import (
    MessageContract,ProgressiveMembraneFusion,split_counts)
from opencood.utils import count4_compact_packet as compact
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.models.fuse_modules.e3dsnn_progressive_sender import ProgressiveSender,l1_partition


class E3dsnnProgressiveDetector(E3dsnnBudget):
    def __init__(self,args):
        if args['activation']!='count4' or args['train_encoder']:
            raise ValueError('Initial progressive interface requires frozen shared Count4 encoder')
        super().__init__(args)
        self.progressive_packets=int(args.get('progressive_packets',3))
        self.readout_prefix=int(args.get('readout_prefix',self.progressive_packets))
        if not 1<=self.readout_prefix<=self.progressive_packets<=3:
            raise ValueError('Registered prefix range is 1..3')
        self.membrane_fusion=ProgressiveMembraneFusion(128,
            retain_residual=bool(args.get('retain_membrane_residual',True)),
            adaptive_gate=bool(args.get('adaptive_membrane_gate',False)))
        self.sender_order=args.get('sender_order','spatial')
        self.fusion_activation_kind='count4'
        self.message_support_kind='received'
        self.sender_ranking='progressive_'+self.sender_order
        if self.sender_order not in ('spatial','frozen','learned','l1'):raise ValueError('Unknown sender order')
        if self.sender_order!='spatial':self.progressive_sender=ProgressiveSender()
        for name,p in self.named_parameters():
            p.requires_grad_(name.startswith('membrane_fusion.basis.') or
                (self.membrane_fusion.adaptive_gate and name.startswith('membrane_fusion.gate.')) or
                (self.sender_order=='learned' and name.startswith('progressive_sender.')))
        self._encoder_identity=None

    def load_state_dict(self,*args,**kwargs):
        result=super().load_state_dict(*args,**kwargs);self._encoder_identity=None
        return result

    def load_legacy(self,state):
        missing,extra=self.load_state_dict(state,strict=False)
        expected={k for k in self.state_dict() if k.startswith(('membrane_fusion.','progressive_sender.'))}
        if set(missing)!=expected or extra:raise ValueError('Unexpected progressive checkpoint mapping')

    def encode_membrane(self,lidar,batch_size):
        captured={}
        def before(module,inputs):captured['precount']=inputs[0]
        handle=self.backbone_3d.conv_out[-1].register_forward_pre_hook(before)
        try:
            sparse=self.encode_lidar(lidar,batch_size)
            membrane=sparse.replace_feature(captured['precount']).dense()
        finally:handle.remove()
        return sparse,membrane

    def _contract(self,data,index,shape):
        if self._encoder_identity is None:self._encoder_identity=encoder_digest(self)
        return MessageContract(str(data['frame_ids'][index]),str(data['road_frame_ids'][index]),
            'infrastructure',self._encoder_identity,'shared-3d-output-v1',tuple(int(n) for n in shape),
            tuple(self.origin),tuple(self.step),tuple(int(x) for x in data['timestamps_us'][index].detach().cpu().tolist()))

    def forward(self,data):
        if self.training and self.sender_order=='l1':raise ValueError('L1 order substitution is an evaluation-only diagnostic')
        if self.training and (self.packet_roundtrip or self.readout_prefix!=self.progressive_packets):
            raise ValueError('Train all prefixes on differentiable messages; wire replay is eval-only')
        b=data['object_bbx_mask'].shape[0]
        if len(data['frame_ids'])!=b or len(data['road_frame_ids'])!=b:
            raise ValueError('Per-sample frame IDs required')
        _,membrane=self.encode_membrane(data['processed_lidar'],b)
        contracts=[self._contract(data,i,membrane.shape[2:]) for i in range(b)]
        states=[self.membrane_fusion.begin(membrane[i:i+1],contracts[i],data['road_to_ego'][i:i+1],
            self._encoder_identity,contracts[i].channel_basis) for i in range(b)]
        ego_features=torch.cat([s.output for s in states]).flatten(1,2)
        if self.fusion_mode=='ego' or data.get('disable_road',False):
            for s in states:self.membrane_fusion.finish(s)
            result=self.detect_bev(ego_features);result['message_kept']=membrane.new_zeros(b)
            return result
        road=self.encode_lidar(data['road_lidar'],b)
        # Keep selection identical on direct and wire paths; per-packet codec uses keep=1.
        from opencood.utils.e3dsnn_budget import budget_features
        values=budget_features(road.indices,road.features,self.activation,self.budget_codec)
        road=road.replace_feature(values).dense()
        partition=split_counts if self.sender_order=='spatial' else self.progressive_sender
        if self.sender_order=='l1':partition=l1_partition
        chunks=[partition(road[i:i+1],self.progressive_packets) for i in range(b)]
        kept=membrane.new_ones(b)
        if self.training and self.message_dropout:
            kept=(torch.rand(b,device=membrane.device)>=self.message_dropout).to(membrane.dtype)
        outputs=[];lengths=[];prefix_lengths=[]
        for k in range(self.readout_prefix):
            for i in range(b):
                part=chunks[i][k]*kept[i];contract=contracts[i];pose=data['road_to_ego'][i:i+1]
                if self.packet_roundtrip:
                    active=(part[0]!=0).any(0);coords=torch.nonzero(active).cpu().numpy()
                    feature=part[0,:,active].T.detach().cpu().numpy()
                    meta=dict(asdict(contract),source_to_ego=pose[0].cpu().tolist(),
                              packet_index=k,packet_count=self.progressive_packets)
                    packet=compact.encode(coords,feature,meta,dict(bits=3,keep_fraction=1.))
                    coords,feature,received=compact.decode(packet,dict(bits=3,keep_fraction=1.))
                    contract=MessageContract(**{name:tuple(received[name]) if name in
                        ('shape_zyx','origin_xyz','step_xyz','timestamps_us') else received[name]
                        for name in asdict(contract)})
                    if received['packet_index']!=k or received['packet_count']!=self.progressive_packets:
                        raise ValueError('Packet sequence metadata mismatch')
                    pose=part.new_tensor(received['source_to_ego'])[None]
                    reconstructed=torch.zeros_like(part);address=torch.as_tensor(coords.astype(np.int64),device=part.device)
                    reconstructed[0,:,address[:,0],address[:,1],address[:,2]]=part.new_tensor(feature.T.copy())
                    part=reconstructed;lengths.append(len(packet))
                self.membrane_fusion.receive(states[i],part,contract,pose,k)
            feature=torch.cat([s.output for s in states]).flatten(1,2)
            outputs.append(self.detect_bev(feature))
            if self.packet_roundtrip:prefix_lengths.append(sum(lengths))
        for s in states:self.membrane_fusion.finish(s)
        result=dict(outputs[-1],message_kept=kept)
        if self.training:
            result['prefix_outputs']=outputs
            if self.enable_ego_aux:result['ego_output']=self.detect_bev(ego_features)
        if self.packet_roundtrip:
            # Existing runner counts packets; report one total per frame, not each fragment.
            result['packet_bytes']=[sum(lengths[i::b]) for i in range(b)]
            result['prefix_total_bytes']=prefix_lengths
        return result
