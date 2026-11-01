"""Frozen mask dependency ablations; no communication byte accounting.

Geometry means strictly positive warped support. Request-only deliberately
uses that geometric support instead of confidence-thresholded edge support.
This probes receiver masking, not a complete replacement transport protocol.
"""
import torch
import torch.nn.functional as F


def geometry_mask(warp, request=None):
    assert warp.ndim==4 and warp.shape[:2]==(2,1)
    assert torch.isfinite(warp).all() and warp.min()>=0 and warp.max()<=1.00001
    mask=(warp>0).to(warp.dtype)
    if request is not None:
        assert request.ndim==4 and request.shape[:2]==(2,1)
        assert torch.isfinite(request).all() and ((request==0)|(request==1)).all()
        resized=F.interpolate(request,size=warp.shape[-2:],mode='nearest')
        # Both agents send to ego0, whose request map applies to the road.
        mask=mask*resized[:1]
    mask[0]=1
    return mask,mask[1:].mean()


class CoSDHSelection:
    def __init__(self,model,mode):
        assert mode in ('identity','request_only','confidence_only','geometry_only')
        self.mode=mode;self.originals=[];self.begin()
        for stage,fusion in enumerate(model.fusion_net):
            assert type(fusion).__name__=='Where2comm' and not fusion.fully
            module=fusion.naive_communication
            assert type(module).__name__=='Communication' and not module.training
            original=module.forward;self.originals.append((module,original))
            module.forward=self.wrap(stage,original)

    def wrap(self,stage,original):
        def forward(confidence,B,warp,req_mask=None):
            assert B==1 and len(confidence)==len(warp)==1 and confidence[0].shape[0]==2
            assert req_mask is not None, 'Registered baseline has an ego request map'
            if self.mode=='identity':result=original(confidence,B,warp,req_mask)
            elif self.mode=='confidence_only':result=original(confidence,B,warp,None)
            else:result=geometry_mask(warp[0],req_mask if self.mode=='request_only' else None)
            mask,rate=result;assert mask.shape==warp[0].shape and (mask[0]==1).all()
            assert torch.isfinite(mask).all() and ((mask==0)|(mask==1)).all()
            self.calls[stage]+=1
            self.rows[stage]=dict(stage=stage,road_retained_cells=int(mask[1].sum()),road_grid_cells=mask[1].numel(),
                geometry_covered_cells=int((warp[0][1]>0).sum()),mask_is_binary=True,ego_mask_all_ones=True)
            return result
        return forward

    def begin(self):self.calls=[0,0,0];self.rows={}

    def finish(self):
        assert self.calls==[1,1,1] and len(self.rows)==3
        return dict(mode=self.mode,calls=self.calls[:],stages=[self.rows[i] for i in range(3)],
            learned_confidence_used=self.mode in ('identity','confidence_only'),
            ego_request_used=self.mode in ('identity','request_only'),
            geometry_policy='original confidence-weighted support' if self.mode in ('identity','confidence_only') else 'positive warped support',
            scope='Receiver mask intervention only; counts and decoder unchanged; no byte or transport advantage claim')

    def close(self):
        for module,original in self.originals:module.forward=original


def self_test():
    warp=torch.tensor([[[[1.,0.],[.2,1.]]],[[[0.,1.],[.3,0.]]]])
    request=torch.tensor([[[[1.,0.],[1.,1.]]],[[[0.,0.],[0.,0.]]]])
    mask,rate=geometry_mask(warp,request)
    assert torch.equal(mask[0],torch.ones_like(mask[0]))
    assert torch.equal(mask[1],torch.tensor([[[0.,0.],[1.,0.]]])) and rate==.25
    request[1]=1
    assert torch.equal(geometry_mask(warp,request)[0],mask),'Road request must not affect ego selection'
    assert geometry_mask(warp)[1]==.5
    request.zero_();assert geometry_mask(warp,request)[1]==0
    assert torch.equal(warp,torch.tensor([[[[1.,0.],[.2,1.]]],[[[0.,1.],[.3,0.]]]]))
    print('Geometric support, ego request, padding and input preservation passed')


if __name__=='__main__':self_test()
