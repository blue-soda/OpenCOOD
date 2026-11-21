"""Fixed-grid exact reuse of each scale's linear contribution to BEV fusion."""
import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionKernel


class BranchFusion(nn.Module):
    def __init__(self, conv, verify=False):
        super().__init__()
        if (conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                conv.groups, conv.dilation) != (384,128,(3,3),(1,1),(1,1),1,(1,1)):
            raise ValueError('Expected the first three-scale BEV fusion convolution')
        self.weight = conv.weight
        self.bias = conv.bias
        self.verify = verify
        self.kernel = NonzeroFusionKernel()
        self.register_buffer('kernels', torch.stack([
            w.detach().permute(2,3,1,0).contiguous() for w in conv.weight.split(128,dim=1)]))

    def begin(self, prior, reason):
        self.previous = None if reason or prior is None else prior['branch_fusion']
        self.new = None

    def forward(self, value):
        if torch.is_grad_enabled() or value.dtype != torch.float32 or value.shape[:2] != (1,384):
            raise ValueError('Frozen FP32 single stream required')
        _,_,h,w = value.shape
        groups = value.split(128,dim=1)
        if self.previous is None:
            dirty = torch.ones(3,1,h,w,device=value.device,dtype=torch.bool)
        else:
            old_value, previous_outputs = self.previous
            changed = (value != old_value).reshape(3,128,h,w).any(dim=1,keepdim=True)
            dirty = F.max_pool2d(changed.float(),3,1,1).bool()
        joint = dirty.any(dim=0).flatten()
        ids = torch.arange(h*w,device=value.device)
        outputs, branches = [], []
        for i, group in enumerate(groups):
            active = dirty[i].flatten()
            selected = ids[active]
            out = (value.new_empty(h*w,128) if self.previous is None else previous_outputs[i].clone())
            actual = self.kernel.run(group,self.kernels[i],selected,out)
            # Each nonzero channel contributes one tap to each output whose
            # receptive field reads it. Padding has zero arithmetic cost.
            counts = group.count_nonzero(dim=1).float().unsqueeze(1)
            taps = (F.avg_pool2d(counts,3,1,1)*9).round().long().flatten()
            expected = int(taps[active].sum())
            assert actual == expected
            if self.verify:
                dense = torch.empty_like(out)
                static = self.kernel.run(group,self.kernels[i],ids,dense)
                assert static == int(taps.sum()) and torch.equal(dense,out)
            branches.append(dict(dirty=len(selected),static_taps=int(taps.sum()),joint_taps=int(taps[joint].sum()),
                                 executed_taps=actual,stable_nonzero_sites=int(((~active)&(taps>0)).sum())))
            outputs.append(out)
        result = (outputs[0]+outputs[1]+outputs[2]).T.reshape(1,128,h,w)
        if self.bias is not None:
            result = result + self.bias.view(1,-1,1,1)
        self.new = (value.detach().clone(),outputs)
        self.ledger = dict(sites=h*w,joint_dirty=int(joint.sum()),branches=branches,
                           float_conv=sum(b['executed_taps'] for b in branches)*256,
                           cache_bytes=(value.numel()+sum(o.numel() for o in outputs))*value.element_size(),
                           merge_add=(2+int(self.bias is not None))*outputs[0].numel())
        if self.verify:
            original = F.conv2d(value,self.weight,self.bias,padding=1)
            err = (result-original).abs()
            self.ledger['max_dense_error'] = float(err.max())
            assert torch.allclose(result,original,atol=2e-4,rtol=2e-5),float(err.max())
        self.previous = None
        return result
