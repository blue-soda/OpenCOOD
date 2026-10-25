"""Same final readout mask, batched native convolutions on bounded tiles.

Only requested cells are written. Expanded/duplicate/edge arithmetic is counted.
Zeroing projected intermediate positions outside the WORLD grid is essential:
BN and biases otherwise synthesize false border features inside padded crops.
"""
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_local_readout import LocalReadoutCache


class TiledReadoutCache(LocalReadoutCache):
    def __init__(self,readout,bounds,threshold,period=3,tile=8,chunk_tiles=32):
        super().__init__(readout,bounds,threshold,period=period)
        if tile not in (8,16) or not isinstance(chunk_tiles,int) or chunk_tiles<1:
            raise ValueError('Only tile8/16 with positive chunk size supported')
        self.tile,self.chunk_tiles=tile,chunk_tiles
        # Native convolutions consume original weights, no packed duplicates.
        self.weights=[]
        self.last_tiles=0

    def selected(self,current,old,mask):
        if torch.is_grad_enabled() or self.readout.training:
            raise ValueError('Frozen inference required')
        if (current.shape[0]!=1 or mask.shape!=(1,1,*current.shape[-2:]) or
                mask.dtype!=torch.bool or old.shape!=(1,self.second.out_channels,*current.shape[-2:])):
            raise ValueError('Matching single-frame input, output cache and mask required')
        if current.device!=old.device or current.dtype!=old.dtype or mask.device!=current.device:
            raise ValueError('Matching device and dtype required')
        h,w=current.shape[-2:];t=self.tile;cells=h*w
        tiles=F.max_pool2d(mask.float(),t,t,ceil_mode=True)>0
        active=tiles.reshape(-1).nonzero(as_tuple=False).flatten()
        self.last_tiles=active.numel()
        output=old.reshape(1,old.shape[1],-1).clone()
        if not active.numel():return output.reshape_as(old),0,0
        source=current[0].permute(1,2,0).reshape(cells,current.shape[1])
        source=torch.cat((source,source.new_zeros(1,source.shape[1])),0)
        flat_mask=torch.cat((mask.flatten(),mask.new_zeros(1)),0)
        outer=torch.arange(-2,t+2,device=current.device)
        middle=torch.arange(-1,t+1,device=current.device)
        core=torch.arange(t,device=current.device)
        for start in range(0,active.numel(),self.chunk_tiles):
            batch=active[start:start+self.chunk_tiles]
            y0=torch.div(batch,tiles.shape[-1],rounding_mode='floor')*t
            x0=(batch%tiles.shape[-1])*t
            def coordinates(offset):
                yy=y0[:,None,None]+offset[None,:,None]
                xx=x0[:,None,None]+offset[None,None,:]
                valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
                return torch.where(valid,yy*w+xx,batch.new_full((),cells)),valid
            indices,_=coordinates(outer)
            patches=source[indices.reshape(-1)].reshape(batch.numel(),t+4,t+4,-1).permute(0,3,1,2).contiguous()
            first=F.conv2d(patches,self.first.weight,self.first.bias)
            first=F.relu(self.bn1(first))
            projected=F.conv2d(first,self.project.weight,self.project.bias)
            _,valid=coordinates(middle)
            projected=projected*valid[:,None].to(projected.dtype)
            result=F.relu(self.bn2(F.conv2d(projected,self.second.weight,self.second.bias)))
            target,valid=coordinates(core)
            write=valid&flat_mask[target]
            rows=result.permute(0,2,3,1)
            output[0,:,target[write]]=rows[write].t()
        support=active.numel()*(t+2)**2
        count=active.numel()*t*t
        macs=support*(self.first.weight.numel()+self.project.weight.numel())+count*self.second.weight.numel()
        return output.reshape_as(old),macs,support
