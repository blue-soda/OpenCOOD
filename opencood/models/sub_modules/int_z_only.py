"""One signed pre-readout spatial state and a separately adaptable reader."""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_y_only import YOnlyUpdater,YOnlyState,YOnlyDetector
from opencood.models.sub_modules.int_anchor_residual import predict


class ZOnlyUpdater(YOnlyUpdater):
    def __init__(self,width=64,context='plain',gate_mode='spatial'):
        if context not in ('plain','dilated'):raise ValueError('Unknown Z context')
        if gate_mode not in ('spatial','channel'):raise ValueError('Unknown Z gate')
        super().__init__(128,96,width)
        self.context,self.gate_mode=context,gate_mode
        if gate_mode=='channel':
            self.gate=nn.Conv2d(width,96,1)
            nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,4.)
        if context=='dilated':
            # Preserve weights/parameter count; enlarge the last hidden layer's neighborhood.
            self.features[4].dilation=(2,2)
            self.features[4].padding=(2,2)

    def load_warm_weights(self,weights):
        weights=dict(weights)
        if self.gate_mode=='channel' and weights['gate.weight'].shape[0]==1:
            weights['gate.weight']=weights['gate.weight'].repeat(96,1,1,1)
            weights['gate.bias']=weights['gate.bias'].repeat(96)
        self.load_state_dict(weights,strict=True)

    def forward(self,evidence,old,valid):
        hidden=self.features(torch.cat((evidence,old,valid.to(old.dtype)),1))
        # Z is before BN/activation and can be negative; never clamp its candidate.
        candidate=self.candidate(hidden)
        weight=.95*torch.sigmoid(self.gate(hidden))*valid.to(old.dtype)
        return weight*old+(1-weight)*candidate


def make_reader(readout,trainable=False):
    reader=nn.Sequential(copy.deepcopy(readout.spatial),copy.deepcopy(readout.tail)).eval()
    for p in reader.parameters():p.requires_grad_(trainable)
    return reader


class ZOnlyDetector(YOnlyDetector):
    def __init__(self,baseline,updater,reader,period=3):
        super().__init__(baseline,updater,period)
        self.reader=reader

    @torch.no_grad()
    def full_from_evidence(self,evidence):
        h,_=self.stage.dense_from_stem(evidence);m=self.model
        second=m.backbone.resnet.layer1(h);third=m.backbone.resnet.layer2(second)
        current=self.readout.current(m.backbone.deblocks[0](h))
        deep=self.readout.history(torch.cat((m.backbone.deblocks[1](second),m.backbone.deblocks[2](third)),1))
        return current+deep

    @torch.no_grad()
    def teacher_y(self,z):
        return self.readout.tail(self.readout.spatial(z))

    def step(self,data,state,meta,supervise=False):
        if torch.is_grad_enabled() and not supervise:raise ValueError('Gradients require explicit supervision')
        evidence,fm,info=self.front(data,state,meta)
        z,new,details=self.cache.step(evidence,None if state is None else state['z'],meta,
            lambda:self.full_from_evidence(evidence),bool(info['reset']),supervise)
        value=self.reader(z);pred=predict(self.model,value);refresh=details['y_refresh']
        info.update({k.replace('y_','z_',1):v for k,v in details.items()})
        if supervise:info.update(y_student=value,y_target=self.teacher_y(info['z_target']))
        info.update(whole_model_macs=self.execution_costs.get(refresh),deep_partial=not refresh,
            spatial_refresh=refresh,spatial_extra_linear_macs=0,spatial_state_bytes=0,
            anchor_bytes=0,readout_state_bytes=z.numel()*z.element_size(),
            readout_refresh=True,readout_write_fraction=1.)
        return pred,dict(fm=fm,z=new),info
