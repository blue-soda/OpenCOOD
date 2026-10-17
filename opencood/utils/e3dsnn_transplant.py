"""Strict state partition: encoder includes BN; receiver includes all heads."""
import hashlib


def compose_state(encoder, receiver):
    if set(encoder) != set(receiver):
        raise ValueError('Donor state keys differ')
    result = {}
    for key in encoder:
        if encoder[key].shape != receiver[key].shape or encoder[key].dtype != receiver[key].dtype:
            raise ValueError('Donor tensor contract differs: ' + key)
        if key.startswith('backbone_3d.'):
            result[key] = encoder[key].clone()
        elif key == 'fusion_scale' or key.startswith(('fusion_net.', 'backbone_2d.', 'cls_head.', 'reg_head.', 'dir_head.')):
            result[key] = receiver[key].clone()
        else:
            raise ValueError('Unclassified state tensor: ' + key)
    return result


def encoder_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.backbone_3d.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
