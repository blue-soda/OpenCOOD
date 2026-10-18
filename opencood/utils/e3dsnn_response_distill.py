"""Anchor-aligned response KD, with frozen teacher and explicit dropout mask.

This utility is preparation only; no existing training path calls it by default.
"""
import torch
import torch.nn.functional as F


def response_distillation(student, teacher, labels, sample_mask, temperature=2.):
    if temperature <= 0:
        raise ValueError('Positive temperature required')
    psm = student['psm']
    if psm.ndim != 4: raise ValueError('Expected [B,A,H,W] logits')
    b,a,h,w = psm.shape
    for key,channels in [('psm',a),('rm',a*7),('dm',a*2)]:
        if student[key].shape != (b,channels,h,w) or teacher[key].shape != student[key].shape:
            raise ValueError('Mismatched anchor response: '+key)
        if not torch.isfinite(student[key]).all() or not torch.isfinite(teacher[key]).all():
            raise FloatingPointError('Nonfinite response: '+key)
    pos,neg = labels['pos_equal_one'],labels['neg_equal_one']
    if pos.shape != (b,h,w,a) or neg.shape != pos.shape:
        raise ValueError('Expected [B,H,W,A] anchor labels')
    if not torch.all((pos==0)|(pos==1)) or not torch.all((neg==0)|(neg==1)) or torch.any((pos>0)&(neg>0)):
        raise ValueError('Invalid or overlapping positive/negative anchors')
    mask = torch.as_tensor(sample_mask,device=psm.device,dtype=psm.dtype).detach()
    if mask.shape != (b,) or not torch.all((mask==0)|(mask==1)):
        raise ValueError('Expected binary [B] road-kept mask')
    pos,neg = pos.reshape(b,-1).to(psm.dtype),neg.reshape(b,-1).to(psm.dtype)

    def flatten(x, codes):
        return x.permute(0,2,3,1).reshape(b,-1,codes)

    def anchor_mean(loss, weights):
        return (loss*weights).sum(1)/weights.sum(1).clamp_min(1)

    def sample_mean(loss):
        # Normalize by batch size, not kept count: dropped messages contribute
        # zero rather than amplifying KD on the remaining samples.
        return (loss*mask).mean()

    s=flatten(student['psm'],1).squeeze(-1)/temperature
    t=flatten(teacher['psm'].detach(),1).squeeze(-1)/temperature
    prob=torch.sigmoid(t)
    kl=prob*(F.logsigmoid(t)-F.logsigmoid(s))+(1-prob)*(F.logsigmoid(-t)-F.logsigmoid(-s))
    cls=sample_mean(.5*(anchor_mean(kl,pos)+anchor_mean(kl,neg)))*temperature**2
    s=flatten(student['rm'],7);t=flatten(teacher['rm'].detach(),7)
    delta=torch.cat((s[...,:6]-t[...,:6],torch.sin(s[...,6:]-t[...,6:])),dim=-1)
    beta=1./9.;absolute=delta.abs()
    reg=sample_mean(anchor_mean(torch.where(absolute<beta,.5*delta**2/beta,absolute-.5*beta).sum(-1),pos))
    s=F.log_softmax(flatten(student['dm'],2)/temperature,dim=-1)
    t=F.log_softmax(flatten(teacher['dm'].detach(),2)/temperature,dim=-1)
    direction=sample_mean(anchor_mean((t.exp()*(t-s)).sum(-1),pos))*temperature**2
    total=cls+2.*reg+.2*direction
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite distillation loss')
    return total,dict(kd_classification=cls,kd_regression=reg,kd_direction=direction)
