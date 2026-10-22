"""Three main controls: residual SNN, jointly trained old IF, residual ANN.

Shallow encoder is frozen; conv4/out, backend and continuous heads train after
interface warmup. All BN buffers stay fixed. Dense reference communication
blocks are not SSC kernels and do not imply neuromorphic speed/energy gains.
"""
from dataclasses import asdict
import hashlib
import json
import numpy as np
import torch
from torch import nn
from opencood.models.e3dsnn_hybrid import E3dsnnHybrid
from opencood.models.fuse_modules.e3dsnn_residual_stream import ResidualSender,ResidualReceiver,select_symbols,masked_symbols
from opencood.models.fuse_modules.e3dsnn_joint_membrane import JointMembraneReceiver
from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire,stream_readout,sample_byte_budget
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume
from opencood.utils import residual_spike_packet as wire


class E3dsnnResidualCoop(E3dsnnHybrid):
    def __init__(self,args):
        super().__init__(dict(args,receiver_activation='relu',freeze_detector=False,freeze_detector_bn=True))
        opts=args['residual_stream'];self.family=opts['family']
        if self.family not in ('residual_snn','old_snn','residual_ann'):raise ValueError('Unknown family')
        if (self.family=='residual_ann') != (self.activation=='relu'):raise ValueError('Encoder family mismatch')
        self.steps=int(opts.get('steps',16));self.event_cap=int(opts.get('event_cap',128))
        if not 1<=self.steps<=255 or self.event_cap<1:raise ValueError('Invalid computation limits')
        self.budget=opts.get('budget_bytes',4096);self.bounds=opts.get('train_budget_range',[2048,8192])
        self.sender=ResidualSender();self.receiver=ResidualReceiver();self.legacy_receiver=JointMembraneReceiver()
        self.rate_weight=float(opts.get('rate_weight',.001))
        self.ego_aux_weight=.5;self.enable_ego_aux=True;self.message_dropout=0.
        self.freeze_encoder=False;self.stage='joint';self.last_packets=[];self.last_records=[]
        self.contract_identity=opts.get('checkpoint_identity','unbound')
        self.set_stage('joint')

    def set_stage(self,stage):
        if stage not in ('interface','joint'):raise ValueError('Unknown stage')
        self.stage=stage
        for name,p in self.named_parameters():
            active=name.startswith('sender.project.')
            if self.family=='old_snn':active=active or name.startswith(('sender.decode.','legacy_receiver.'))
            else:active=active or name.startswith(('sender.','receiver.'))
            if stage=='joint':active=active or name.startswith(('backbone_3d.conv4.','backbone_3d.conv_out.','backbone_2d.','cls_head.','reg_head.','dir_head.'))
            p.requires_grad_(active)

    def train(self,mode=True):
        nn.Module.train(self,mode)
        for module in self.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def encode_membrane(self,lidar,batch):
        capture={}
        handle=self.backbone_3d.conv_out[-1].register_forward_pre_hook(lambda m,x:capture.update(value=x[0]))
        try:
            sparse=self.encode_lidar(lidar,batch)
            potential=sparse.replace_feature(capture['value']).dense()
        finally:handle.remove()
        return sparse,potential

    def metadata(self,data,i,shape):
        contract=MessageContract(str(data['frame_ids'][i]),str(data['road_frame_ids'][i]),'infrastructure',
            self.contract_identity,'residual32-'+self.contract_identity,tuple(shape),tuple(self.origin),tuple(self.step),
            tuple(int(x) for x in data['timestamps_us'][i].cpu().tolist()))
        meta=dict(contract=asdict(contract),source_to_ego=data['road_to_ego'][i].cpu().tolist(),
                  channels=32,shape_zyx=list(shape),steps=self.steps,family=self.family,schema='ordered-residual-v1')
        return contract,meta

    def generate(self,source,metadata,budget):
        events=[];packet=b'';chunks=[];rate=[]
        if self.family=='old_snn':
            drive=torch.relu(self.sender.project(source));stream=integrate_and_fire(drive,8)
            candidate=[(min(self.steps-1,int(t*self.steps)),int(a),1) for t,a in zip(stream.times,stream.addresses)]
            events,packet,n,_=wire.append_prefix([],candidate,metadata,budget)
            counts=stream.prefix(n)
            prob=1-torch.exp(-drive.abs())
            proxy=(6*prob.sum()+4*prob.amax(1).sum())/drive.numel()
            return ([(0,counts)] if n else []),packet,events,proxy
        reconstruction=torch.zeros_like(source)
        potential=source.new_zeros((1,32)+source.shape[2:])
        for step in range(self.steps):
            symbols,value=self.sender.step(source,reconstruction,potential,self.family)
            candidate=select_symbols(symbols,value.abs(),self.event_cap,step)
            events,packet,n,stopped=wire.append_prefix(events,candidate,metadata,budget)
            chosen=candidate[:n];selected=masked_symbols(symbols,chosen)
            # Costs are an explicitly biased smooth proxy, never byte reporting.
            prob=1-torch.exp(-value.abs())
            rate.append((6*prob.sum()+4*prob.amax(1).sum())/value.numel())
            if self.family=='residual_snn':potential=value-selected
            reconstruction=reconstruction+self.sender.decode(selected)
            if chosen:chunks.append((step,selected))
            if stopped:break
        proxy=torch.stack(rate).mean() if rate else source.sum()*0
        return chunks,packet,events,proxy

    def decode_chunks(self,packet,metadata,device_tensor):
        events=wire.decode(packet,metadata);chunks=[]
        if not len(events):return chunks
        if self.family=='old_snn':
            arr=np.bincount(events[:,1],weights=events[:,2],minlength=32*int(np.prod(metadata['shape_zyx'])))
            return [(0,device_tensor.new_tensor(arr).reshape((1,32)+tuple(metadata['shape_zyx'])))]
        for step in np.unique(events[:,0]):
            rows=events[events[:,0]==step]
            q=device_tensor.new_zeros((1,32)+tuple(metadata['shape_zyx']))
            q.reshape(-1)[torch.as_tensor(rows[:,1],device=q.device)]=q.new_tensor(rows[:,2])
            chunks.append((int(step),q))
        return chunks

    def fuse(self,local,ego,chunks,pose,contract):
        if not chunks:return ego
        if self.family=='old_snn':
            return stream_readout(local,chunks[0][1],pose,contract,self.legacy_receiver,self.sender.decode)
        reconstruction=torch.zeros_like(local);support=local[:,:1]*0
        state=self.receiver.begin(local);correction=torch.zeros_like(local)
        for _,q in chunks:
            reconstruction=reconstruction+self.sender.decode(q)
            support=torch.maximum(support,(q.detach()!=0).any(1,keepdim=True).to(q.dtype))
            road,coverage=warp_volume(reconstruction,pose,self.origin,self.step,geometry_dtype=torch.float64)
            mask,_=warp_volume(support,pose,self.origin,self.step,geometry_dtype=torch.float64)
            mask=mask.clamp(0,1)*(coverage>0).to(mask.dtype)
            correction=self.receiver.receive(state,local,road,mask,self.family)
        return ego+correction

    def forward(self,data):
        b=data['object_bbx_mask'].shape[0]
        sparse,membrane=self.encode_membrane(data['processed_lidar'],b);ego=sparse.dense()
        local_only=self.fusion_mode=='ego' or data.get('disable_road',False) or (not self.training and self.budget==0)
        self.last_packets=[];self.last_records=[]
        if local_only:
            return dict(self.detect_bev(ego.flatten(1,2)),packet_bytes=[0]*b,message_kept=ego.new_zeros(b))
        road_sparse,road_membrane=self.encode_membrane(data['road_lidar'],b)
        source=road_sparse.dense() if self.family=='old_snn' else road_membrane
        features=[];sizes=[];rates=[];budgets=[];counts=[]
        for i in range(b):
            budget=sample_byte_budget(self.bounds,'log_uniform') if self.training else self.budget
            contract,metadata=self.metadata(data,i,ego.shape[2:])
            chunks,packet,events,rate=self.generate(source[i:i+1],metadata,budget)
            if self.packet_roundtrip:
                if self.training:raise ValueError('Wire-only reconstruction is evaluation-only')
                decoded=self.decode_chunks(packet,metadata,ego)
                if len(decoded)!=len(chunks) or any(a!=c or not torch.equal(x,y) for (a,x),(c,y) in zip(chunks,decoded)):
                    raise ValueError('Direct/wire step mismatch')
                chunks=decoded
            features.append(self.fuse(membrane[i:i+1],ego[i:i+1],chunks,data['road_to_ego'][i:i+1],contract))
            sizes.append(len(packet));rates.append(rate);counts.append(len(events));budgets.append(budget)
            self.last_packets.append(packet);self.last_records.append(dict(frame_id=contract.receiver_frame,
                actual_bytes=len(packet),events=len(events),steps=len(chunks),metadata=metadata,
                sha256=hashlib.sha256(packet).hexdigest()))
        out=dict(self.detect_bev(torch.cat(features).flatten(1,2)),packet_bytes=sizes,stream_events=counts,
                 stream_budgets=budgets,message_kept=ego.new_tensor([bool(n) for n in counts]))
        if self.training:
            out['ego_output']=self.detect_bev(ego.flatten(1,2))
            out['message_rate_loss']=self.rate_weight*torch.stack(rates).mean()
        return out
