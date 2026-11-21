"""Compact ordinary middle update with exact two-layer spatial support."""
import torch
import torch.nn.functional as F


def dilate(mask):
    return F.max_pool2d(mask.float(),3,1,1).bool()


def supports(active):
    first=dilate(active)
    return active,first,dilate(first)


def conv_costs(core):
    enc=core.coarse_light
    upd=core.coarse_update
    res=upd.candidate.residual
    mods={'radius2':(enc[0],enc[2],upd.candidate.history,res[0]),
          'radius1':(res[2],),'output':(res[4],res[6],upd.gate[0],upd.gate[2])}
    return {k:sum(2*m.weight.numel() for m in v) for k,v in mods.items()}


def points(x):
    return x[0].flatten(1).T


def pw(module,x):
    return F.linear(x,module.weight[:,:,0,0],module.bias)


def lookup(indices, size):
    lut=indices.new_full((size,),len(indices))
    lut[indices]=torch.arange(len(indices),device=indices.device)
    return lut


def neighbors(ids,out_w,in_shape,stride):
    h,w=in_shape
    axis=torch.arange(-1,2,device=ids.device)
    yy=ids.div(out_w,rounding_mode='floor')[:,None,None]*stride+axis[None,:,None]
    xx=(ids%out_w)[:,None,None]*stride+axis[None,None,:]
    valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
    return (yy*w+xx).clamp(0,h*w-1).flatten(1),valid.flatten(1)


def patches(values,ids,out_w,shape,stride,lut=None):
    idx,valid=neighbors(ids,out_w,shape,stride)
    if lut is not None:idx=lut[idx]
    assert not bool(((idx==len(values))&valid).any())
    idx=idx.masked_fill(~valid,len(values))
    return F.pad(values,(0,0,0,1))[idx].transpose(1,2)


def spatial_conv(module,x,ids,width,shape,lut):
    neighborhood=patches(x,ids,width,shape,1,lut)
    return F.linear(neighborhood.reshape(len(ids),-1),module.weight.flatten(1),module.bias)


def compact_update(core,fine,old,valid,dt,active,held):
    a,b,c=supports(active)
    h,w=a.shape[-2:]
    ids,one,two=[s.flatten().nonzero().flatten() for s in (a,b,c)]
    map1,map2=lookup(one,h*w),lookup(two,h*w)
    enc=core.coarse_light
    upd=core.coarse_update
    res=upd.candidate.residual
    patch=patches(points(fine),two,w,fine.shape[-2:],2)
    dw=(patch*enc[0].weight.reshape(1,-1,9)).sum(2)
    if enc[0].bias is not None:dw+=enc[0].bias
    observed=pw(enc[2],dw.relu())
    valid2=points(valid)[two]
    old2=points(old)[two].masked_fill(~valid2,0.)
    x=observed+pw(upd.candidate.history,torch.cat((old2,valid2.to(old2.dtype)),1))
    h0=pw(res[0],x).relu()
    h1=spatial_conv(res[2],h0,one,w,(h,w),map2).relu()
    h2=spatial_conv(res[4],h1,ids,w,(h,w),map1).relu()
    selected=map2[ids]
    candidate=(x[selected]+pw(res[6],h2)).relu()
    older,visible=old2[selected],valid2[selected]
    clock=torch.full_like(visible,dt,dtype=x.dtype)
    gate=torch.cat((observed[selected],older,visible.to(x.dtype),clock),1)
    z=pw(upd.gate[2],pw(upd.gate[0],gate).relu()).sigmoid()
    result=held.clone()
    result[0].flatten(1)[:,ids]=torch.where(visible,older+z*(candidate-older),candidate).T
    return result,dict(output=len(ids),radius1=len(one),radius2=len(two))
