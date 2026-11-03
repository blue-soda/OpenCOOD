"""One actual local BN/ReLU chain, with explicit extra output state.

Only fine block1 BN1 and its first ReLU are localized. Convolution output
decode, residual addition, BN2, final ReLU and other blocks remain dense.
"""
import torch
from torch import nn
from opencood.models.sub_modules.int_fine_integer import FineIntegerDetector
from opencood.models.sub_modules.int_fine_support_audit import output_support
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason


def local_bn_relu(x, bn, previous=None, support=None):
    if torch.is_grad_enabled() or bn.training or x.ndim!=4 or x.shape[0]!=1:
        raise ValueError('Frozen single stream required')
    if previous is None:
        y=torch.relu(bn(x));count=x.numel();positions=x.shape[-2]*x.shape[-1];indices=0
    else:
        if previous.shape!=x.shape or previous.dtype!=x.dtype or previous.device!=x.device:
            raise ValueError('Output state changed without reset')
        if support is None or support.dtype!=torch.bool or tuple(support.shape)!=(1,1,*x.shape[-2:]):
            raise ValueError('Spatial support required')
        index=torch.nonzero(support.reshape(-1),as_tuple=False).reshape(-1)
        positions=index.numel();count=positions*x.shape[1];indices=index.numel()*index.element_size()
        y=previous.clone()
        if positions:
            selected=x.reshape(1,x.shape[1],-1).index_select(2,index).unsqueeze(2)
            values=torch.relu(bn(selected))
            y.reshape(1,x.shape[1],-1).index_copy_(2,index,values.squeeze(2))
    return y,dict(total_elements=x.numel(),processed_bn_elements=count,processed_relu_elements=count,
                  processed_positions=positions,cold=previous is None,output_cache_bytes=y.numel()*y.element_size(),
                  index_bytes=indices,gathered_input_bytes=0 if previous is None else count*x.element_size(),
                  boundary='Counts only block1 BN1/first ReLU. Output clone/scatter, support generation, dense current Conv decode and all residual operations still charged as work; no FLOPs/latency inference.')


class LocalPointwiseBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        for name in ('conv1','bn1','relu','conv2','bn2','downsample'):setattr(self,name,getattr(block,name))
        self.stride=block.stride;self.previous_output=None;self.next_output=None;self.pointwise_statistics={}

    def forward(self,x):
        identity=x;out=self.conv1(x);leaf=self.conv1
        if leaf.mode!='delta' or leaf.next_state is None:raise ValueError('Delta first convolution required')
        if (leaf.previous is None)!=(self.previous_output is None):raise ValueError('Coupled n/A and pointwise reset required')
        support=None
        if leaf.previous is not None:
            delta=leaf.next_state[0].to(torch.int64)-leaf.previous[0].to(torch.int64)
            support=output_support(delta,leaf.source)
        out,self.pointwise_statistics=local_bn_relu(out,self.bn1,self.previous_output,support)
        self.next_output=out
        out=self.bn2(self.conv2(out))
        if self.downsample is not None:identity=self.downsample(x)
        out+=identity
        return self.relu(out)


class FineLocalPointwiseDetector(FineIntegerDetector):
    def __init__(self,source,calibration,capture_audit=False):
        super().__init__(source,calibration,'hybrid',capture_audit)
        self.core.observation.layers[1]=LocalPointwiseBlock(self.core.observation.layers[1])
        self.variant='fine_local_bn1_relu';self.eval()

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        block=self.core.observation.layers[1];reason,_=reset_reason(state,meta)
        block.previous_output=None if reason else state['fine_pointwise']
        try:
            pred,new,info=super().step_spatial(spatial,state,meta,detach_state,requests)
            new['fine_pointwise']=block.next_output;info['fine_pointwise']=dict(block.pointwise_statistics)
            info['cache_bytes']+=block.pointwise_statistics['output_cache_bytes']
            return pred,new,info
        finally:block.previous_output=None;block.next_output=None
