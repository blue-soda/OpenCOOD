"""One encoder pass and geometric-only receiver, without confidence/request.

Reuses the checkpoint's parameterized layers and generic warp/max interface;
local membrane and IF readout hooks remain attached to those interfaces.
No call to the source model forward, confidence decoder or Communication.
"""
import torch


class GeometryBackend:
    def __init__(self,model):
        from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
        from opencood.utils.transformation_utils import normalize_pairwise_tfm
        self.warp=warp_affine_simple;self.normalize=normalize_pairwise_tfm
        assert type(model.backbone).__name__=='BaseBEVBackbone' and len(model.fusion_net)==3
        cfg=model.backbone.model_cfg
        assert list(cfg['layer_strides'])==[2,2,2] and list(cfg['upsample_strides'])==[1,2,4]
        assert model.compression and model.shrink_flag and not model.training
        # All shrink convolutions preserve the first stage spatial grid.
        for module in model.shrink_conv.modules():
            if isinstance(module,torch.nn.Conv2d):
                assert module.stride==(1,1) and module.dilation==(1,1)
                assert tuple(2*p+1 for p in module.padding)==module.kernel_size
        self.model=model;self.original_model=model.forward;self.original_fusions=[];self.handles=[]
        self.begin()
        def forbidden(module,inputs):raise AssertionError('Removed confidence/request route executed')
        self.handles.append(model.backbone.register_forward_pre_hook(forbidden))
        self.handles.append(model.simple_scatter.register_forward_pre_hook(forbidden))
        self.handles.append(model.cls_head.register_forward_hook(self.head_hook))
        for stage,fusion in enumerate(model.fusion_net):
            assert not fusion.fully and type(fusion.fuse_modules).__name__=='MaxFusion'
            self.handles.append(fusion.naive_communication.register_forward_pre_hook(forbidden))
            self.original_fusions.append((fusion,fusion.forward))
            fusion.forward=self.fusion_forward(stage,fusion)
        model.forward=self.forward

    def head_hook(self,module,inputs,output):
        self.head_calls+=1
        assert output.shape[0]==1 and output.shape[-2:]==self.canvas_shape

    def fusion_forward(self,stage,fusion):
        def forward(x,record_len,transform):
            assert record_len.tolist()==[2] and x.shape[0]==2
            h,w=x.shape[-2:];matrix=transform[0,0,:2]
            aligned=self.warp(x,matrix,(h,w))
            # Preserve the exact legacy geometric support sampling grid, which
            # was the confidence map's grid, without computing confidence.
            support=self.warp(x.new_ones((2,1)+tuple(self.canvas_shape)),matrix,(h,w))
            mask=(support>0).to(x.dtype);mask[0]=1
            aligned=aligned*mask
            self.fusion_calls[stage]+=1
            self.rows[stage]=dict(stage=stage,road_retained_cells=int(mask[1].sum()),road_grid_cells=mask[1].numel())
            value=fusion.fuse_modules(aligned,record_len,transform,use_warp_feature=False)
            return value,mask[1:].mean()
        return forward

    def forward(self,data):
        m=self.model;assert data['record_len'].tolist()==[2]
        assert data.get('_ego_flag',True) and not m.training
        lidar=data['processed_lidar']
        batch=dict(voxel_features=lidar['voxel_features'],voxel_coords=lidar['voxel_coords'],
            voxel_num_points=lidar['voxel_num_points'],record_len=data['record_len'])
        batch=m.scatter(m.pillar_vfe(batch));spatial=batch['spatial_features']
        transform=self.normalize(data['pairwise_t_matrix'],spatial.shape[-2],spatial.shape[-1],m.voxel_size[0])
        features=m.backbone.get_multiscale_feature(spatial);assert len(features)==3
        self.canvas_shape=features[0].shape[-2:]
        fused=[]
        for stage,feature in enumerate(features):
            decoded=m.naive_compressor_list[stage](feature,use_fp16=True)
            value,_=m.fusion_net[stage](decoded,data['record_len'],transform)
            fused.append(value)
        bev=m.shrink_conv(m.backbone.decode_multiscale_feature(fused))
        result=dict(cls_preds=m.cls_head(bev),reg_preds=m.reg_head(bev))
        if m.use_dir:result['dir_preds']=m.dir_head(bev)
        self.model_calls+=1
        return result

    def begin(self):
        self.model_calls=0;self.head_calls=0;self.fusion_calls=[0,0,0];self.rows={};self.canvas_shape=None

    def finish(self):
        assert self.model_calls==self.head_calls==1 and self.fusion_calls==[1]*3
        return dict(model_calls=1,cls_head_calls=1,fusion_calls=self.fusion_calls[:],encoder_paths=1,
            confidence_backbone_forward_calls=0,request_scatter_calls=0,original_communication_calls=0,
            confidence_canvas_shape=list(self.canvas_shape),stages=[self.rows[i] for i in range(3)],
            scope='Single shared encoder; geometry warp, local membrane and IF readout; ANN decoder/head retained')

    def close(self):
        self.model.forward=self.original_model
        for fusion,original in self.original_fusions:fusion.forward=original
        for h in self.handles:h.remove()
