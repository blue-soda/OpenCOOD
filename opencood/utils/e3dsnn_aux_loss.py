"""Matched auxiliary-supervision controls, without changing inference."""
import torch


def detection_loss(output, labels, criterion, weight=0.):
    prefixes = output.get('prefix_outputs')
    if prefixes is not None:
        if not prefixes:raise ValueError('Nonempty progressive prefixes required')
        values=[]
        for prefix in prefixes:
            for key in ('psm','rm','dm'):
                if prefix[key].shape!=output[key].shape or not torch.isfinite(prefix[key]).all():
                    raise ValueError('Invalid progressive detection output: '+key)
            values.append(criterion(prefix,labels))
        fused=torch.stack(values).mean()
    else:
        fused = criterion(output, labels)
    if not torch.isfinite(fused):
        raise FloatingPointError('Nonfinite fused detection loss')
    parts = {'fused_detection_loss': float(fused.detach())}
    if prefixes is not None:
        parts.update({'prefix_{}_detection_loss'.format(i+1):float(v.detach()) for i,v in enumerate(values)})
    if 'ego_output' in output:
        for key in ('psm', 'rm', 'dm'):
            if output['ego_output'][key].shape != output[key].shape or not torch.isfinite(output['ego_output'][key]).all():
                raise ValueError('Invalid auxiliary detection output: ' + key)
        ego = criterion(output['ego_output'], labels)
        if not torch.isfinite(ego):
            raise FloatingPointError('Nonfinite ego detection loss')
        parts['ego_detection_loss'] = float(ego.detach())
        return fused + weight * ego, parts
    if weight:
        raise ValueError('Nonzero auxiliary weight requires an ego output')
    return fused, parts
