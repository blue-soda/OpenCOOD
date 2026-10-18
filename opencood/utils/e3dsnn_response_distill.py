"""Anchor-aligned response KD, with frozen teacher and explicit dropout mask.

Enabled only by an explicit response_distillation training configuration.
"""
import torch
import torch.nn.functional as F


def add_response_supervision(loss, parts, output, labels, teacher, data, weight):
    """Teacher stays outside student modules/optimizer/checkpoints, including KD=0."""
    if weight not in (0., 1.):
        raise ValueError('Registered response KD controls are weight 0 or 1')
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Teacher must be frozen and in eval mode')
    with torch.no_grad():
        target = teacher(data)
    kd, components = response_distillation(output, target, labels, output['message_kept'])
    parts.update({k: float(v.detach()) for k, v in components.items()})
    parts.update(kd_loss=float(kd.detach()), kd_weight=weight,
                 message_kept_fraction=float(output['message_kept'].float().mean()))
    return loss if weight == 0. else loss + kd


def state_digest(model):
    import hashlib
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def load_teacher(student_config, device):
    """Validate data/anchor identity, pin source hashes and preserve RNG state."""
    import hashlib
    from pathlib import Path
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils
    spec = student_config['response_distillation']
    if spec['weight'] not in (0., 1.): raise ValueError('Unregistered KD weight')
    for name in ('config', 'checkpoint'):
        if hashlib.sha256(Path(spec[name]).read_bytes()).hexdigest() != spec[name+'_sha256']:
            raise ValueError('Teacher source hash mismatch: '+name)
    config = load_yaml(spec['config'])
    for key in ('fusion', 'preprocess', 'postprocess', 'root_dir', 'validate_dir', 'data_manifest', 'data_augment'):
        if config[key] != student_config[key]: raise ValueError('Teacher/student contract differs: '+key)
    options = config['model']['args']
    if (config['model']['core_method'] != 'e3dsnn_hybrid' or
            options['activation'] != 'relu' or options['receiver_activation'] != 'relu' or
            options['fusion_mode'] != 'residual' or options.get('message_codec') is not None):
        raise ValueError('Expected full-message ANN hybrid teacher')
    with torch.random.fork_rng():
        teacher = train_utils.create_model(config).to(device)
        teacher.load_state_dict(torch.load(spec['checkpoint'], map_location='cpu'), strict=True)
    teacher.requires_grad_(False); teacher.eval()
    return teacher, dict(spec, state_sha256=state_digest(teacher))


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
