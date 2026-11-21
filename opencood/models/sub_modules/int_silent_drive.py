"""A pointwise predictor of the middle recurrence at newly silent valid cells."""
import numpy as np
import torch


def samples(fine, old, ids):
    grid = fine[0,::16,::2,::2]
    if grid.shape != (8,*old.shape[-2:]):
        raise ValueError('Expected eight channels at phase-zero middle centers')
    return old[0].flatten(1)[:,ids].T, grid.flatten(1)[:,ids].T


class DriveFit:
    def __init__(self):
        self.by_scene = {}

    def observe(self, scene, fine, old, extra, full):
        ids = extra.flatten().nonzero().flatten()[::16]
        if not len(ids):
            return
        state, shallow = samples(fine,old,ids)
        state, shallow = state.double(),shallow.double()
        base = torch.cat((torch.ones_like(shallow[:,:1]),shallow),dim=1)
        target = full[0].flatten(1)[:,ids].T.double()-state
        # Shared eight-channel inputs and a separate old value per output channel.
        batch = dict(base=base.T@base,old_base=state.T@base,old_sq=state.square().sum(0),
                     response_base=target.T@base,old_response=(state*target).sum(0),
                     response_sq=target.square().sum(0),count=len(ids))
        if scene not in self.by_scene:
            self.by_scene[scene]=batch
        else:
            for k,value in batch.items():
                self.by_scene[scene][k]+=value

    def finish(self):
        scenes=list(self.by_scene)
        parts={}
        for name,s in self.by_scene.items():
            gram=np.zeros((256,10,10))
            gram[:,0,0]=s['old_sq'].cpu().numpy()
            gram[:,1:,1:]=s['base'].cpu().numpy()
            gram[:,0,1:]=gram[:,1:,0]=s['old_base'].cpu().numpy()
            rhs=torch.cat((s['old_response'][:,None],s['response_base']),1).cpu().numpy()
            parts[name]=(gram,rhs,float(s['response_sq'].sum()),s['count'])
        def fit(names,size):
            g=sum(parts[n][0][:,:size,:size] for n in names)
            h=sum(parts[n][1][:,:size] for n in names)
            reg=.001*np.maximum(np.diagonal(g,axis1=1,axis2=2),1e-8)
            idx=np.arange(size)
            g[:,idx,idx]+=reg
            return np.linalg.solve(g,h[...,None])[...,0]
        folds,coeff={},{}
        for mode,size in [('old',2),('drive',10)]:
            fold={}
            for name in scenes:
                g,h,norm,count=parts[name]
                w=fit([v for v in scenes if v!=name],size)
                loss=norm-2*(w*h[:,:size]).sum()+np.einsum('ci,cij,cj->',w,g[:,:size,:size],w)
                fold[name]=dict(hold_sse=norm,predicted_sse=float(loss),cells=count)
            folds[mode]=fold
            coeff[mode]=fit(scenes,size).tolist()
        return dict(coefficients=coeff,folds=folds,ridge=.001,sampling_stride=16)


class SilentDrive:
    def __init__(self, coefficients, mode):
        self.mode=mode
        self.weight=torch.as_tensor(coefficients[mode],device='cuda',dtype=torch.float32)
        if self.weight.shape != (256,2 if mode=='old' else 10):
            raise ValueError('Invalid drive coefficients')

    def apply(self, fine, old, extra, output):
        ids=extra.flatten().nonzero().flatten()
        if not len(ids):
            return 0
        state=old[0].flatten(1)[:,ids].T
        delta=state*self.weight[:,0]+self.weight[:,1]
        if self.mode=='drive':
            shallow=fine[0,::16,::2,::2].flatten(1)[:,ids].T
            delta=torch.addmm(delta,shallow,self.weight[:,2:].T)
        output[0].flatten(1)[:,ids]=(state+delta).clamp_min(0).T
        # Old slope+bias and residual addition, plus eight multiply-adds for drive.
        return len(ids)*256*(3 if self.mode=='old' else 19)
