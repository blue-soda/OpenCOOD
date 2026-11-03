"""The fixed event pilot objective, independent of dataset and training driver."""
import torch

READOUT_DENSE_MAC = 20230963200
UNIT_ENERGY_RATIO = .1
EVENT_WEIGHT = .1


def window_objective(detection_losses, unit_costs, penalized):
    if not detection_losses or not unit_costs:
        raise ValueError('Original eligible labelled window and all-frame costs required')
    detection = torch.stack(detection_losses).mean()
    mean_units = torch.stack(unit_costs).mean()
    event = UNIT_ENERGY_RATIO * mean_units / READOUT_DENSE_MAC
    # Keep the control's backward graph identical to the original R-221 graph.
    total = detection + EVENT_WEIGHT * event if penalized else detection
    return total, detection, event


def gradient_summary(names, gradients):
    if len(names) != len(gradients):
        raise ValueError('Gradient names must match')
    values = {}
    for name, grad in zip(names, gradients):
        if grad is None:
            values[name] = dict(present=False, finite=True, norm=0., nonzero=0)
        else:
            values[name] = dict(present=True, finite=bool(torch.isfinite(grad).all()),
                                norm=float(grad.detach().double().norm()),
                                nonzero=int(torch.count_nonzero(grad)))
    return dict(parameters=values, finite=all(x['finite'] for x in values.values()),
                nonzero_tensors=sum(x['nonzero'] > 0 for x in values.values()),
                norm=sum(x['norm'] ** 2 for x in values.values()) ** .5)
