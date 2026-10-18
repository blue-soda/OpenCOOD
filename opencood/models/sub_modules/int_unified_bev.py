"""One recurrent BEV grid with bounded local updates and direct detection readout."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev


class LocalStateUpdate(nn.Module):
    """Radius two on the packed grid; no spatial normalization or global attention."""
    radius = 2

    def __init__(self, evidence=128, channels=128):
        super().__init__()
        self.evidence=evidence;self.channels=channels
        self.net=nn.Sequential(
            nn.Conv2d(evidence+channels+1,channels,1),nn.ReLU(),
            nn.Conv2d(channels,channels,3,padding=1,groups=channels),nn.ReLU(),
            nn.Conv2d(channels,channels,1),nn.ReLU(),
            nn.Conv2d(channels,channels,3,padding=1,groups=channels),nn.ReLU(),
            nn.Conv2d(channels,channels*2,1))
        self.current=nn.Conv2d(evidence,channels,1)

    def forward(self,packed):
        old=packed[:,self.evidence:self.evidence+self.channels]
        valid=packed[:,-1:]
        residual,gate=self.net(packed).chunk(2,dim=1)
        candidate=self.current(packed[:,:self.evidence])+residual
        # Newly visible cells must initialize from current evidence, not zero history.
        retain=gate.sigmoid()*valid
        return retain*old+(1-retain)*candidate


class UnifiedBEVCore(nn.Module):
    channels=128
    memory_policy='recurrent'

    def __init__(self, cls_channels=2, reg_channels=14, dir_channels=4):
        super().__init__()
        self.observation=nn.Sequential(nn.Conv2d(64,96,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(96,96,3,padding=1,groups=96),nn.ReLU(),nn.Conv2d(96,96,1),nn.ReLU())
        self.context=nn.Sequential(nn.Conv2d(96,32,1),nn.ReLU(),
            nn.Conv2d(32,32,5,padding=2,groups=32),nn.ReLU())
        self.update=LocalStateUpdate()
        self.readout=nn.Sequential(nn.Conv2d(128,256,1),nn.ReLU())
        self.cls_head=nn.Conv2d(256,cls_channels,1)
        self.reg_head=nn.Conv2d(256,reg_channels,1)
        self.dir_head=nn.Conv2d(256,dir_channels,1) if dir_channels else None

    def evidence(self,spatial):
        observation=self.observation(spatial)
        coarse=F.avg_pool2d(observation,4,4,ceil_mode=True,count_include_pad=False)
        context=F.interpolate(self.context(coarse),size=observation.shape[-2:],mode='nearest')
        return torch.cat([observation,context],dim=1)

    def predict(self,value):
        features=self.readout(value)
        result=dict(psm=self.cls_head(features),rm=self.reg_head(features))
        if self.dir_head is not None:result['dm']=self.dir_head(features)
        return result

    def step(self,spatial,state,meta,bounds,detach_state=True):
        evidence=self.evidence(spatial)
        if self.memory_policy=='reset':state=None
        reason='initial' if state is None else None
        if meta.get('reset_before',False):reason='explicit'
        if state is not None and reason is None:
            if state['scene']!=meta['segment']:reason='scene'
            else:
                gap=(int(meta['timestamp_us'])-state['timestamp_us'])/1e6
                if gap<=0:raise ValueError('Non-causal unified state')
                if gap>.25:reason='gap'
        if reason is not None:
            old=evidence.new_zeros((1,self.channels,*evidence.shape[-2:]))
            valid=torch.zeros_like(old[:,:1],dtype=torch.bool)
        else:
            old,valid=warp_bev(state['value'],state['pose'],meta['pose'],bounds)
        value=self.update(torch.cat([evidence,old,valid.to(old.dtype)],dim=1))
        prediction=self.predict(value)
        new=dict(value=value.detach() if detach_state else value,pose=meta['pose'],
                 scene=meta['segment'],timestamp_us=int(meta['timestamp_us']))
        size=value.numel()*value.element_size()
        info=dict(reset=reason,cache_bytes=size,state_bytes=size,
                  memory_policy=self.memory_policy,
                  unified_dense_update=True,original_backbone_called=False,
                  cache_stages=[dict(actual_write_fraction=0.,executed_input_area_ratio=0.) for _ in range(3)])
        return prediction,new,info


class ChannelNorm(nn.Module):
    """Normalize channels at each cell, without coupling different spatial cells."""
    def __init__(self,channels):
        super().__init__()
        self.norm=nn.LayerNorm(channels)

    def forward(self,x):
        return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()


class SpatialResidual(nn.Module):
    """Dense spatial mixing inside a bottleneck; finite dependency radius."""
    def __init__(self,channels,hidden,dilation=1):
        super().__init__()
        self.net=nn.Sequential(ChannelNorm(channels),nn.Conv2d(channels,hidden,1),nn.SiLU(),
            nn.Conv2d(hidden,hidden,3,padding=dilation,dilation=dilation),nn.SiLU(),
            nn.Conv2d(hidden,channels,1))

    def forward(self,x):
        return x+self.net(x)


class ResidualStateUpdate(nn.Module):
    """V2 update remains radius two on precomputed current evidence plus history."""
    radius=2

    def __init__(self,evidence=256,channels=256):
        super().__init__()
        self.evidence=evidence;self.channels=channels
        self.net=nn.Sequential(nn.Conv2d(evidence+channels+1,channels,1),ChannelNorm(channels),nn.SiLU(),
            SpatialResidual(channels,channels//2),SpatialResidual(channels,channels//2),
            ChannelNorm(channels),nn.Conv2d(channels,channels*2,1))
        self.current=nn.Conv2d(evidence,channels,1)

    def forward(self,packed):
        old=packed[:,self.evidence:self.evidence+self.channels]
        valid=packed[:,-1:]
        residual,gate=self.net(packed).chunk(2,dim=1)
        candidate=self.current(packed[:,:self.evidence])+residual
        retain=gate.sigmoid()*valid
        return retain*old+(1-retain)*candidate


class UnifiedBEVCoreV2(UnifiedBEVCore):
    """Higher-capacity evidence encoder; one state and no multiscale decoder."""
    channels=256

    def __init__(self,cls_channels=2,reg_channels=14,dir_channels=4,memory_policy='recurrent'):
        nn.Module.__init__(self)
        if memory_policy not in ('recurrent','reset'):raise ValueError('Unknown memory policy')
        self.memory_policy=memory_policy
        self.observation=nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1),ChannelNorm(128),nn.SiLU(),
            SpatialResidual(128,64),SpatialResidual(128,64))
        self.context=nn.Sequential(nn.Conv2d(128,192,3,stride=2,padding=1),ChannelNorm(192),nn.SiLU(),
            SpatialResidual(192,96),nn.Conv2d(192,256,3,stride=2,padding=1),ChannelNorm(256),nn.SiLU(),
            SpatialResidual(256,128,1),SpatialResidual(256,128,2),SpatialResidual(256,128,3),
            nn.Conv2d(256,128,1))
        self.update=ResidualStateUpdate()
        self.readout=nn.Sequential(ChannelNorm(256),nn.Conv2d(256,256,1),nn.SiLU())
        self.cls_head=nn.Conv2d(256,cls_channels,1)
        self.reg_head=nn.Conv2d(256,reg_channels,1)
        self.dir_head=nn.Conv2d(256,dir_channels,1) if dir_channels else None

    def evidence(self,spatial):
        observation=self.observation(spatial)
        context=F.interpolate(self.context(observation),size=observation.shape[-2:],mode='bilinear',align_corners=False)
        return torch.cat([observation,context],dim=1)


class UnifiedBEVDetector(nn.Module):
    """Own only the frozen Pillar encoder and new core; no teacher or old backbone."""
    def __init__(self,teacher,config,variant='v1',memory_policy='recurrent'):
        super().__init__()
        if variant not in ('v1','v2'):raise ValueError('Unknown unified architecture')
        if variant=='v1' and memory_policy!='recurrent':raise ValueError('V1 contract is recurrent only')
        self.variant=variant;self.memory_policy=memory_policy
        self.pillar_vfe=copy.deepcopy(teacher.pillar_vfe)
        self.scatter=copy.deepcopy(teacher.scatter)
        self.bounds=config['model']['args']['lidar_range']
        heads=(teacher.cls_head.out_channels,teacher.reg_head.out_channels,
               teacher.dir_head.out_channels if teacher.use_dir else 0)
        self.core=UnifiedBEVCore(*heads) if variant=='v1' else UnifiedBEVCoreV2(*heads,memory_policy=memory_policy)
        self.core.cls_head.load_state_dict(teacher.cls_head.state_dict())
        self.core.reg_head.load_state_dict(teacher.reg_head.state_dict())
        if teacher.use_dir:self.core.dir_head.load_state_dict(teacher.dir_head.state_dict())
        for p in self.pillar_vfe.parameters():p.requires_grad_(False)
        self.train(False)

    def train(self,mode=True):
        super().train(mode)
        self.pillar_vfe.eval();self.scatter.eval()
        return self

    def step(self,data,state,meta,detach_state=True):
        p=data['processed_lidar']
        with torch.no_grad():
            if p['voxel_features'].shape[0]:
                batch={k:p[k] for k in ['voxel_features','voxel_coords','voxel_num_points']}
                batch['batch_size']=1
                spatial=self.scatter(self.pillar_vfe(batch))['spatial_features']
            else:
                nx,ny,_=self.scatter.model_cfg['grid_size']
                spatial=self.core.cls_head.weight.new_zeros((1,64,int(ny),int(nx)))
        return self.core.step(spatial,state,meta,self.bounds,detach_state)
