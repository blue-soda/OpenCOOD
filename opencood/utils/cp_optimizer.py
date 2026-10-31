"""Explicit new-module learning rate and one-epoch LR warmup for CP studies."""
import torch


def make_optimizer(cfg, model):
    from opencood.tools import train_utils
    opts = cfg['optimizer']
    prefix = opts.get('new_parameter_prefix')
    if not prefix:
        return train_utils.setup_optimizer(cfg, model)
    base, new = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (new if name.startswith(prefix) else base).append(parameter)
    if not base or not new or opts['new_parameter_lr'] <= 0:
        raise ValueError('Expected disjoint, nonempty pretrained and new parameter groups')
    method = getattr(torch.optim, opts['core_method'])
    return method([dict(params=base, lr=opts['lr'], group_name='pretrained'),
                   dict(params=new, lr=opts['new_parameter_lr'], group_name='new')],
                  lr=opts['lr'], **opts.get('args', {}))


def apply_warmup(optimizer, epoch, batch, batches, warmup_epochs):
    if warmup_epochs not in (0, 1):
        raise ValueError('Registered warmup is zero or one epoch')
    if warmup_epochs and epoch == 0:
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * (batch + 1) / batches
