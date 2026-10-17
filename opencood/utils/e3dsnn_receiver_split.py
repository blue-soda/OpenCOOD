"""Split adapted receivers while requiring exactly the same sender state."""
import torch


def compose_receiver_state(snn, ann, fusion, bev):
    if fusion not in ('s', 'a') or bev not in ('s', 'a'):
        raise ValueError('Unknown receiver donor')
    if set(snn) != set(ann):
        raise ValueError('Donor state keys differ')
    donors = {'s': snn, 'a': ann}
    result = {}
    for key, value in snn.items():
        other = ann[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError('Incompatible donor tensor: ' + key)
        if key.startswith('backbone_3d.'):
            if not torch.equal(value, other):
                raise ValueError('Encoder donors must be identical: ' + key)
            result[key] = value.clone()
        elif key == 'fusion_scale' or key.startswith('fusion_net.'):
            result[key] = donors[fusion][key].clone()
        elif key.startswith(('backbone_2d.', 'cls_head.', 'reg_head.', 'dir_head.')):
            result[key] = donors[bev][key].clone()
        else:
            raise ValueError('Unclassified state tensor: ' + key)
    return result
