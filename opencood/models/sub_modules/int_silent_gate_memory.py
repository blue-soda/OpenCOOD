"""Reuse targets and gates obtained from real full middle updates."""
import torch

from opencood.models.sub_modules.int_feature_memory import warp_bev


class SilentGateMemory:
    def __init__(self,mode,audit):
        self.mode,self.audit=mode,audit
        self.reset()

    def reset(self):
        self.data=self.valid=self.pose=self.observation=None

    def capture(self,ids,gate,candidate):
        self.observation=ids,gate,candidate

    def prepare(self,old,quiet,area,pose):
        if self.valid is None:
            b,c,h,w=old.shape
            self.data=old.new_zeros((b,2*c,h,w))
            self.valid=torch.zeros_like(quiet)
        else:
            self.data=warp_bev(self.data,self.pose,pose,area)[0]
            self.valid=warp_bev(self.valid.float(),self.pose,pose,area)[0].bool()
        self.pose=pose
        self.covered=quiet & self.valid
        return quiet & ~self.covered

    def update(self,old,visible,output):
        ids=self.covered.flatten().nonzero().flatten()
        channels=old.shape[1]
        targets,gates=self.data[0].flatten(1).split(channels,0)
        alternate=ids.roll(len(ids)//2) if self.mode!='relax' else ids
        g=gates[:,alternate if self.mode=='shuffle_gate' else ids]
        candidate=targets[:,alternate if self.mode=='shuffle_target' else ids]
        prior=old[0].flatten(1)[:,ids]
        output[0].flatten(1)[:,ids]=prior+g*(candidate-prior)
        if self.audit:
            self.reference=torch.lerp(prior,candidate,g)
            self.ids=ids

        full,new_gate,new_target=self.observation
        use=visible.flatten()[full].bool()
        anchors=full[use]
        new_gate,new_target=new_gate[use].T,new_target[use].T
        if self.audit:
            prev=old[0].flatten(1)[:,anchors]
            assert torch.allclose(torch.lerp(prev,new_target,new_gate),output[0].flatten(1)[:,anchors],atol=1e-4,rtol=1e-5)
        targets[:,anchors]=new_target
        gates[:,anchors]=new_gate
        self.valid.flatten()[anchors]=True
        self.observation=None
        return dict(covered=len(ids),anchors=len(anchors),gate_ops=3*channels*len(ids),
                    state_bytes=self.data.numel()*self.data.element_size()+self.valid.numel()*self.valid.element_size())
