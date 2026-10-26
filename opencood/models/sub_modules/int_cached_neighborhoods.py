"""Bounded geometry-only 3x3 lookup; no frame activations or masks retained."""
import torch


class CachedNeighborhoods:
    def __init__(self):
        self.key=None
        self.table=None

    @property
    def geometry_bytes(self):
        return 0 if self.table is None else self.table.numel()*self.table.element_size()

    def __call__(self,mask):
        if mask.ndim!=4 or mask.shape[:2]!=(1,1) or mask.dtype!=torch.bool:
            raise ValueError('Expected single boolean spatial mask')
        height,width=mask.shape[-2:]
        if height<1 or width<1:raise ValueError('Empty spatial grid')
        key=(height,width,mask.device)
        if key!=self.key:
            cells=height*width
            positions=torch.arange(cells,device=mask.device)
            offsets=torch.arange(-1,2,device=mask.device)
            yy=torch.div(positions[:,None,None],width,rounding_mode='floor')+offsets[None,:,None]
            xx=positions[:,None,None]%width+offsets[None,None,:]
            valid=(yy>=0)&(yy<height)&(xx>=0)&(xx<width)
            self.table=torch.where(valid,yy*width+xx,positions.new_full((),cells)).reshape(cells,9)
            self.key=key
        selected=mask.reshape(-1).nonzero(as_tuple=False).flatten()
        return selected,self.table[selected],tuple(mask.shape)
