"""Explicit new-module learning rate and one-epoch LR warmup for CP studies."""
import torch


def make_optimizer(cfg, model):
    from opencood.tools import train_utils
    opts = cfg['optimizer']
    prefix = opts.get('new_parameter_prefix')
    if not prefix:
        return train_utils.setup_optimizer(cfg, model)
    if isinstance(prefix, list):
        prefix = tuple(prefix)
    base, new = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (new if name.startswith(prefix) else base).append(parameter)
    if not base or not new or opts['new_parameter_lr'] <= 0:
        raise ValueError('Expected disjoint, nonempty pretrained and new parameter groups')
    method = getattr(torch.optim, opts['core_method'])
    new_group = dict(params=new, lr=opts['new_parameter_lr'], group_name='new')
    if 'new_parameter_weight_decay' in opts:
        decay = float(opts['new_parameter_weight_decay'])
        if not 0 <= decay < float('inf'):
            raise ValueError('New-parameter weight decay must be finite and nonnegative')
        new_group['weight_decay'] = decay
    return method([dict(params=base, lr=opts['lr'], group_name='pretrained'), new_group],
                  lr=opts['lr'], **opts.get('args', {}))


def new_parameter_health(optimizer):
    """Opt-in cold-start health, independent of communication accounting."""
    groups = [g for g in optimizer.param_groups if g.get('group_name') == 'new']
    if len(groups) != 1:
        raise ValueError('Expected one explicit new-parameter group')
    norms = torch.stack([p.detach().norm() for p in groups[0]['params']])
    values = norms.cpu().tolist()
    return dict(parameter_l2=float(norms.norm()), minimum_tensor_l2=min(values),
                tensors_below_1e_8=sum(v < 1e-8 for v in values), tensors=len(values),
                weight_decay=groups[0]['weight_decay'], learning_rate=groups[0]['lr'])


def apply_warmup(optimizer, epoch, batch, batches, warmup_epochs):
    if warmup_epochs not in (0, 1):
        raise ValueError('Registered warmup is zero or one epoch')
    if warmup_epochs and epoch == 0:
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * (batch + 1) / batches
