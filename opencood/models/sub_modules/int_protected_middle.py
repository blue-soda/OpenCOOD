"""Allocate full silent-cell updates while exactly matching both convolution halos."""
import numpy as np
import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_compact_middle_update import supports
from opencood.models.sub_modules.int_feature_memory import warp_bev


def sparse_returns(lidar, spatial_shape):
    h,w = ((n+3)//4 for n in spatial_shape)
    coords = lidar['voxel_coords'].long()
    y,x = (coords[:,-2]+2)//4,(coords[:,-1]+2)//4
    valid = (y<h)&(x<w)
    grid = torch.zeros(h*w,device=coords.device,dtype=torch.float)
    grid.scatter_add_(0,(y*w+x)[valid],lidar['voxel_num_points'][valid].float().flatten())
    count = F.avg_pool2d(grid.view(1,1,h,w),3,1,1)*9
    return (count.round()>=1)&(count.round()<=20)


def aligned_confidence(previous, meta, spatial_shape, bounds):
    score = previous['guard_confidence']
    area = block_lidar_range(bounds,spatial_shape,score.shape[-2:],2)
    aligned,_ = warp_bev(score,previous['pose'],meta['pose'],area)
    return F.max_pool2d(aligned,5,2,2)


def protected_mask(base, silent, score, frame):
    """The 2-cell halos of different 4x4 cores have disjoint 8x8 ownership."""
    h,w = silent.shape[-2:]
    top,left = (3*frame)%8,(5*frame)%8
    bottom,right = (-(h+top))%8,(-(w+left))%8
    pad=(left,right,top,bottom)
    y=torch.arange(h,device=silent.device)+top
    x=torch.arange(w,device=silent.device)+left
    inner=(y%8>=2)&(y%8<6)
    inner=inner[:,None] & ((x%8>=2)&(x%8<6))[None,:]
    proposals=silent & inner
    active=base & ~silent
    def tiles(v):
        padded=F.pad(v.float(),pad)
        return padded.reshape((h+top+bottom)//8,8,(w+left+right)//8,8)
    counts=[]
    for expanded,old in zip(supports(active|proposals),supports(active)):
        counts.append(tiles(expanded & ~old).sum((1,3)).int().flatten())
    sig=torch.stack(counts,1).cpu().numpy()
    scores=tiles(score.masked_fill(~proposals,0)).amax((1,3)).flatten().cpu().numpy()
    ids=np.arange(len(sig),dtype=np.int64)
    # Mix the frame number before hashing tile IDs, so orders vary between frames.
    tie=((ids ^ ((frame*2246822519)&0xffffffff))*2654435761)&0xffffffff
    restored=np.zeros(len(sig),dtype=bool)
    exchangeable=0
    for key in np.unique(sig[sig[:,0]>0],axis=0):
        bucket=ids[(sig==key).all(axis=1)]
        parity=(int(key@[11,17,29])+frame)%2
        n=(len(bucket)+parity)//2
        exchangeable+=int(key[0])*min(n,len(bucket)-n)
        order=np.lexsort((tie[bucket],-scores[bucket]))
        restored[bucket[order[:n]]]=True
    tile_mask=torch.as_tensor(restored,device=silent.device).view((h+top+bottom)//8,-1)
    chosen=tile_mask.repeat_interleave(8,0).repeat_interleave(8,1)[top:top+h,left:left+w] & proposals
    expected=sig[restored].sum(0).tolist()
    return chosen,dict(increments=expected,available=int(proposals.sum()),blocks=int(restored.sum()),
                       exchangeable=exchangeable)


class ProtectedMiddle:
    def __init__(self, mode):
        self.mode=mode

    def select(self, base, silent, innovation, frame):
        if self.mode=='random':
            score=torch.zeros_like(innovation)
        elif self.mode=='innovation':
            score=F.max_pool2d(innovation,5,1,2)
        elif self.mode=='sparse':
            score=self.sparse.float()
        elif self.mode=='prior':
            score=self.prior
        else:
            score=self.prior*(1+self.sparse.float())+.1*self.sparse.float()
        self.chosen,info=protected_mask(base,silent,score,int(frame))
        return silent & ~self.chosen,info
