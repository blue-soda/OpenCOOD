"""Auditable reference CoSDH transport, not the authors' supplied codec.

Sender uses ego's transmitted demand bitmap, its own confidence and shared pose.
It sends target selection masks plus FP16 latent sites required by decoder/warp,
and confidence-filtered local 7D boxes+scores. Receiver fusion and NMS consume
decoded packets. Diagnostic redundant computation is retained for parity checks;
do not use this implementation to claim optimized compute/latency.
"""
import hashlib,struct,time,zlib
from pathlib import Path
import numpy as np
from infocom_wire import envelope,ENVELOPE

HEADER=struct.Struct('<4sBBHIIIIII')

def pack(kind,stage,h,w,c,n,raw,flag=0):
    body=zlib.compress(raw,6)
    return HEADER.pack(b'CDW1',1,kind,stage|(flag<<8),h,w,c,n,len(body),len(raw))+body

def unpack(packet):
    magic,version,kind,tag,h,w,c,n,size,rawsize=HEADER.unpack_from(packet)
    if magic!=b'CDW1' or version!=1 or len(packet)!=HEADER.size+size:raise ValueError('Bad CoSDH packet')
    raw=zlib.decompress(packet[HEADER.size:])
    if len(raw)!=rawsize:raise ValueError('Bad body size')
    return kind,tag&255,tag>>8,h,w,c,n,raw

def encode_request(mask):
    mask=np.asarray(mask,dtype=np.float32)
    if mask.ndim!=2 or not np.all((mask==0)|(mask==1)):raise ValueError('Binary demand required')
    h,w=mask.shape;return pack(0,0,h,w,1,int(mask.sum()),np.packbits(mask.astype(np.uint8).reshape(-1)).tobytes())

def decode_request(packet):
    kind,stage,flag,h,w,c,n,raw=unpack(packet);assert kind==0 and c==1
    assert len(raw)==(h*w+7)//8
    mask=np.unpackbits(np.frombuffer(raw,dtype=np.uint8))[:h*w].astype(np.float32).reshape(h,w)
    assert int(mask.sum())==n
    return mask

def encode_latent(stage,latent,support,target_mask):
    latent=np.asarray(latent,dtype=np.float32);support=np.asarray(support,dtype=bool);target_mask=np.asarray(target_mask,dtype=np.float32)
    c,h,w=latent.shape;assert support.shape==target_mask.shape==(h,w)
    if not np.array_equal(latent.astype(np.float16).astype(np.float32),latent):raise ValueError('Source is not exact FP16')
    selected=latent.transpose(1,2,0)[support].astype('<f2')
    binary=np.all((target_mask==0)|(target_mask==1))
    target=np.packbits(target_mask.astype(np.uint8).reshape(-1)).tobytes() if binary else target_mask.astype('<f4').tobytes()
    raw=np.packbits(support.reshape(-1)).tobytes()+selected.tobytes()+target
    return pack(1,stage,h,w,c,int(support.sum()),raw,0 if binary else 1)

def decode_latent(packet):
    kind,stage,flag,h,w,c,n,raw=unpack(packet);assert kind==1 and flag in (0,1)
    bitmap_size=(h*w+7)//8
    support=np.unpackbits(np.frombuffer(raw[:bitmap_size],dtype=np.uint8))[:h*w].astype(bool)
    assert support.sum()==n
    end=bitmap_size+n*c*2
    selected=np.frombuffer(raw[bitmap_size:end],dtype='<f2').astype(np.float32).reshape(n,c)
    latent=np.zeros((h*w,c),np.float32);latent[support]=selected
    if flag==0:
        assert len(raw)==end+bitmap_size
        target=np.unpackbits(np.frombuffer(raw[end:],dtype=np.uint8))[:h*w].astype(np.float32).reshape(h,w)
    else:
        assert len(raw)==end+h*w*4
        target=np.frombuffer(raw[end:],dtype='<f4').copy().reshape(h,w)
    return stage,latent.reshape(h,w,c).transpose(2,0,1).copy(),target

def encode_boxes(boxes_and_scores,transform=None):
    x=np.asarray(boxes_and_scores,dtype='<f4');assert x.ndim==2 and x.shape[1]==8
    extra=b''
    if transform is not None:
        transform=np.asarray(transform,dtype='<f4');assert transform.shape==(4,4)
        extra=transform.tobytes()
    return pack(2,0,len(x),8,1,len(x),x.tobytes()+extra,1 if transform is not None else 0)

def decode_boxes(packet):
    kind,stage,flag,h,w,c,n,raw=unpack(packet);assert kind==2 and w==8 and n==h and flag in (0,1) and len(raw)==h*8*4+flag*64
    boxes=np.frombuffer(raw[:h*8*4],dtype='<f4').copy().reshape(h,8)
    return (boxes,np.frombuffer(raw[h*8*4:],dtype='<f4').copy().reshape(4,4)) if flag else boxes

class CoSDHWire:
    def __init__(self,model,dataset,output):
        assert len(model.fusion_net)==3 and not any(m.fully for m in model.fusion_net), 'Audited masked three-scale config required'
        self.model=model;self.dataset=dataset;self.output=Path(output)/'wire';self.output.mkdir()
        self.handles=[];self.rebuilding=False;self.cache={}
        for stage in range(3):
            self.handles += [model.naive_compressor_list[stage].decoder.register_forward_pre_hook(self.capture_latent(stage)),
                             model.fusion_net[stage].register_forward_pre_hook(self.capture_fusion(stage)),
                             model.fusion_net[stage].naive_communication.register_forward_pre_hook(self.request_hook),
                             model.fusion_net[stage].naive_communication.register_forward_hook(self.capture_mask(stage)),
                             model.fusion_net[stage].fuse_modules.register_forward_pre_hook(self.receive_hook(stage))]
        self.original_post=dataset.post_processor.post_process
        dataset.post_processor.post_process=self.post_process

    def begin(self,batch,frame_id):
        assert batch['ego']['record_len'].tolist()==[2], 'Matched two-agent audit only'
        self.batch=batch;self.frame_id=frame_id;self.cache={};self.parts={};self.details=[];self.request=None
        self.pose=batch['ego']['pairwise_t_matrix'][0,0,1].detach().cpu().numpy()

    def capture_latent(self,stage):
        def hook(module,inputs):
            if not self.rebuilding and inputs[0].shape[0]==2:self.cache[(stage,'latent')]=inputs[0].detach().clone()
        return hook

    def capture_fusion(self,stage):
        def hook(module,inputs):
            assert inputs[0].shape[0]==2
            self.cache[(stage,'matrix')]=inputs[3][0,0,1:2].detach().clone()
        return hook

    def request_hook(self,module,inputs):
        import torch
        req=inputs[3];assert req is not None
        mask=req[0,0].detach().cpu().numpy()
        packet=encode_request(mask);decoded=decode_request(packet)
        assert np.array_equal(mask,decoded)
        if self.request is None:self.request=packet
        else:assert self.request==packet, 'Demand must be reused, not counted three times'
        restored=req.clone();restored[0,0]=torch.from_numpy(decoded).to(req.device)
        return inputs[:3]+(restored,)

    def capture_mask(self,stage):
        def hook(module,inputs,output):self.cache[(stage,'mask')]=output[0][1:2].detach().clone()
        return hook

    def receive_hook(self,stage):
        def hook(module,inputs):
            import torch
            import torch.nn.functional as F
            from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
            started=time.perf_counter();full=inputs[0];h,w=full.shape[-2:]
            latent=self.cache[(stage,'latent')];matrix=self.cache[(stage,'matrix')];mask=self.cache[(stage,'mask')]
            if mask.shape[-2:]!=(h,w):mask=F.interpolate(mask,size=(h,w),mode='bilinear',align_corners=False)
            active=mask[0,0]!=0
            grid=F.affine_grid(matrix,(1,full.shape[1],h,w),align_corners=False)[0]
            x0=torch.floor(((grid[...,0][active]+1)*w-1)/2).long();y0=torch.floor(((grid[...,1][active]+1)*h-1)/2).long()
            source=torch.zeros((h,w),device=full.device)
            for dx in (0,1):
                for dy in (0,1):
                    x=x0+dx;y=y0+dy;ok=(x>=0)&(x<w)&(y>=0)&(y<h);source[y[ok],x[ok]]=1
            support=F.max_pool2d(source[None,None],7,1,3)[0,0].bool()
            packet=encode_latent(stage,latent[1].cpu().numpy(),support.cpu().numpy(),mask[0,0].cpu().numpy())
            decoded_stage,received,received_mask=decode_latent(packet);assert decoded_stage==stage
            sparse=latent.clone();sparse[1]=torch.from_numpy(received).to(latent.device)
            self.rebuilding=True
            try:decoded=self.model.naive_compressor_list[stage].decoder(sparse)
            finally:self.rebuilding=False
            road=warp_affine_simple(decoded[1:2],matrix,(h,w))*torch.from_numpy(received_mask).to(full.device)[None,None]
            error=float((full[1:2]-road).abs().max());exact=torch.equal(full[1:2],road)
            if error>1e-5:raise ValueError('Transport changed retained feature: '+str(error))
            replacement=full.clone();replacement[1:2]=road
            self.parts[stage]=packet
            self.details.append({'stage':stage,'selected_target_sites':int(active.sum()),'source_latent_sites':int(support.sum()),'section_bytes':len(packet),'feature_exact':exact,'max_abs_feature_error':error,'reference_transport_seconds':time.perf_counter()-started})
            return (replacement,)+inputs[1:]
        return hook

    def post_process(self,data_dict,output_dict,confidence_beta=None,confidence_threshold=None):
        """Official operations/order, with remote 7D boxes+scores serialized before projection."""
        import torch
        from opencood.utils import box_utils
        from opencood.utils.common_utils import limit_period
        pp=self.dataset.post_processor;corners_list=[];score_list=[]
        for cav_id,content in data_dict.items():
            output=output_dict[cav_id]
            assert 'iou_preds' not in output and 'psm' not in output
            prob=torch.sigmoid(output['cls_preds'].permute(0,2,3,1))
            if confidence_beta is not None and confidence_threshold is not None and cav_id!='ego':
                prob[prob<confidence_threshold]=0.;prob*=confidence_beta
            prob=prob.reshape(1,-1);reg=output['reg_preds'];assert len(reg.shape)==4
            boxes=pp.delta_to_boxes3d(reg,content['anchor_box']);select=(prob>pp.params['target_args']['score_threshold']).view(1,-1)
            boxes=torch.masked_select(boxes[0],select.unsqueeze(2).repeat(1,1,7)[0]).view(-1,7)
            scores=torch.masked_select(prob[0],select[0])
            if 'dir_preds' in output and len(boxes):
                offset=pp.params['dir_args']['dir_offset'];bins=pp.params['dir_args']['num_bins']
                direction=output['dir_preds'].permute(0,2,3,1).contiguous().reshape(1,-1,bins)[select].max(dim=-1)[1]
                period=2*np.pi/bins
                rotation=limit_period(boxes[...,6]-offset,0,period)
                boxes[...,6]=rotation+offset+period*direction.to(boxes.dtype)
                boxes[...,6]=limit_period(boxes[...,6],.5,2*np.pi)
            transform=content['transformation_matrix']
            if cav_id!='ego':
                source=torch.cat((boxes,scores[:,None]),1).detach().cpu().numpy();source_pose=transform.detach().cpu().numpy()
                packet=encode_boxes(source,source_pose);received,received_pose=decode_boxes(packet)
                assert np.array_equal(source,received)
                assert np.array_equal(source_pose,received_pose)
                transform=torch.from_numpy(received_pose).to(reg.device)
                self.parts[3]=packet;boxes=torch.from_numpy(received[:,:7].copy()).to(reg.device);scores=torch.from_numpy(received[:,7].copy()).to(reg.device)
                self.late_count=len(boxes)
            if len(boxes):
                corners=box_utils.boxes_to_corners_3d(boxes,order=pp.params['order'])
                projected=box_utils.project_box3d(corners,transform)
                # Retain exact official construction/order prior to global NMS.
                standup=box_utils.corner_to_standup_box_torch(projected)
                corners_list.append(projected);score_list.append(torch.cat((standup,scores[:,None]),1))
        if not corners_list:return None,None
        pred=torch.vstack(corners_list);scores=torch.vstack(score_list)[:,-1]
        keep=torch.logical_and(box_utils.remove_large_pred_bbx(pred),box_utils.remove_bbx_abnormal_z(pred));pred=pred[keep];scores=scores[keep]
        keep=box_utils.nms_rotated(pred,scores,pp.params['nms_thresh']);pred=pred[keep];scores=scores[keep]
        filtered,keep=box_utils.mask_boxes_outside_range_numpy(pred.cpu().numpy(),pp.params['gt_range'],order=None,return_mask=True)
        return torch.from_numpy(filtered).to(pred.device),scores[keep]

    def finish(self,index):
        assert sorted(self.parts)==[0,1,2,3] and self.request is not None
        backward=envelope(self.frame_id,0,self.pose)+self.request
        forward=envelope(self.frame_id,1,self.pose)+struct.pack('<4sI',b'CFW1',4)+b''.join(self.parts[i] for i in range(4))
        rows=[]
        for name,packet in [('demand',backward),('response',forward)]:
            p=self.output/('%06d_%s.bin'%(index,name));p.write_bytes(packet)
            rows.append({'direction':name,'file':'wire/'+p.name,'bytes':len(packet),'sha256':hashlib.sha256(packet).hexdigest()})
        return {'packets':rows,'demand_bytes':len(backward),'response_bytes':len(forward),'total_bytes':len(backward)+len(forward),'envelope_bytes':160,'late_boxes':self.late_count,'sections':self.details,'late_section_bytes':len(self.parts[3])}

    def close(self):
        for handle in self.handles:handle.remove()
        self.dataset.post_processor.post_process=self.original_post

def self_test():
    rng=np.random.RandomState(303);latent=rng.randn(4,13,17).astype(np.float16).astype(np.float32);support=rng.rand(13,17)>.7
    for mask in [(rng.rand(13,17)>.5).astype(np.float32),rng.rand(13,17).astype(np.float32)]:
        packet=encode_latent(1,latent,support,mask);stage,got,target=decode_latent(packet)
        assert stage==1 and np.array_equal(got,latent*support) and np.array_equal(target,mask)
    request=(rng.rand(13,17)>.5).astype(np.float32);assert np.array_equal(request,decode_request(encode_request(request)))
    for n in (0,7):
        boxes=rng.randn(n,8).astype(np.float32);assert np.array_equal(boxes,decode_boxes(encode_boxes(boxes)))
        received,pose=decode_boxes(encode_boxes(boxes,np.eye(4,dtype=np.float32)));assert np.array_equal(received,boxes) and np.array_equal(pose,np.eye(4))
    print('CoSDH demand, sparse latent/binary or continuous mask, empty/nonempty box codec round trips passed')

if __name__=='__main__':self_test()
