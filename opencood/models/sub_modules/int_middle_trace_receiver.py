"""Transport the last observed middle-update error or its packed sign."""
import torch

from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_silent_drive import samples


def pack_directions(values):
    """Pack four channel signs per byte; input shape is (channels, sites)."""
    lanes = torch.where(values > 0, 1, torch.where(values < 0, 2, 0)).byte()
    a,b,c,d = lanes.reshape(-1,4,values.shape[1]).unbind(1)
    return a | (b << 2) | (c << 4) | (d << 6)


def unpack_directions(data):
    shifts = torch.arange(0,8,2,device=data.device,dtype=torch.uint8)
    lanes = (data[:,None] >> shifts[None,:,None]) & 3
    return ((lanes == 1).float() - (lanes == 2).float()).flatten(0,1)


class MiddleTraceReceiver:
    def __init__(self, mode, readout, verify=False):
        self.mode = mode
        self.bias,self.scale = torch.tensor(readout[mode],device='cuda',dtype=torch.float32)[:,:,None]
        self.verify = verify
        self.reset()

    def reset(self):
        self.cache = self.valid = self.reference = self.pose = None

    def align(self, value, past, current, area):
        return warp_bev(value.float(),past,current,area)[0].to(value.dtype)

    def apply(self, fine, old, valid, full_sites, quiet, output, weight, area, pose):
        channels = old.shape[1]
        packed = self.mode in ('sign','shuffled')
        if self.valid is None:
            known = torch.zeros_like(valid,dtype=torch.bool)
            if self.mode!='bias':
                data = old.new_zeros((1,channels//4 if packed else channels,*old.shape[-2:]),
                                     dtype=torch.uint8 if packed else old.dtype)
                if self.verify:self.reference = torch.zeros_like(old)
        else:
            known = self.align(self.valid,self.pose,pose,area)
            if self.mode!='bias':
                data = self.align(self.cache,self.pose,pose,area)
                if self.verify:self.reference = self.align(self.reference,self.pose,pose,area)

        ids = (quiet & known).flatten().nonzero().flatten()
        source = ids.roll(len(ids)//2) if self.mode=='shuffled' else ids
        adjustment = self.bias
        if self.mode!='bias' and len(ids):
            value = data[0].flatten(1)[:,source]
            value = unpack_directions(value) if packed else value
            if self.verify:
                expected=self.reference[0].flatten(1)[:,source]
                assert torch.equal(value,expected.sign() if packed else expected)
            contribution = (torch.where(value>0,self.scale,torch.where(value<0,-self.scale,0.))
                            if packed else value*self.scale)
            adjustment = self.bias+contribution
        if len(ids):
            flat=output[0].flatten(1)
            flat[:,ids]=(flat[:,ids]+adjustment).clamp_min(0)
        if self.verify:
            self.audit_ids,self.audit_adjustment=ids,adjustment

        observed=full_sites & valid.bool()
        anchors=observed.flatten().nonzero().flatten()
        if self.mode!='bias':
            if len(anchors):
                state,shallow=samples(fine,old,anchors)
                delta=torch.addmm(state*weight[:,0]+weight[:,1],shallow,weight[:,2:].T)
                residual=output[0].flatten(1)[:,anchors]-(state+delta).clamp_min(0).T
                data[0].flatten(1)[:,anchors]=pack_directions(residual) if packed else residual
                if self.verify:self.reference[0].flatten(1)[:,anchors]=residual
            self.cache=data
        self.valid,self.pose=known | observed,pose
        cache_bytes=self.valid.numel()*self.valid.element_size()
        if self.cache is not None:cache_bytes+=self.cache.numel()*self.cache.element_size()
        # Each recorded channel evaluates the 19-op drive and subtracts it from the actual update.
        record_ops=0 if self.mode=='bias' else 20*channels*len(anchors)
        read_ops=channels*len(ids)*({'bias':1,'sign':2,'shuffled':2,'amplitude':3}[self.mode])
        return dict(anchors=len(anchors),covered=len(ids),cache_bytes=cache_bytes,record_ops=record_ops,read_ops=read_ops)
