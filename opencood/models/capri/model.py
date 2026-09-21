"""Per-instance recurrent prediction and sparse current-time keypoint fusion."""
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import associate, move_keypoints, transform_boxes, transform_points, register_background


class Capri(nn.Module):
    def __init__(self, feature_dim, settings):
        super().__init__()
        self.settings = settings
        hidden = settings['hidden_dim']
        self.instance_encoder = nn.Sequential(nn.Linear(feature_dim+9, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.recurrent = nn.GRUCell(hidden, hidden)
        self.motion = nn.Sequential(nn.Linear(hidden+4, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.point_encoder = nn.Sequential(nn.Linear(feature_dim+3, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.query = nn.Linear(hidden, hidden)
        self.refiner = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, 8))
        nn.init.zeros_(self.motion[-1].weight)
        nn.init.zeros_(self.motion[-1].bias)
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def encode(self, message, confidence):
        valid = message['valid'].to(message['features'].dtype)
        pooled = (message['features'] * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        boxes = message['boxes']
        geometry = torch.cat((boxes[:, :3]/100, boxes[:, 3:6]/10,
                              boxes[:, 6:7].sin(), boxes[:, 6:7].cos(),
                              boxes.new_full((len(boxes), 1), confidence)), 1)
        return self.instance_encoder(torch.cat((pooled, geometry), 1))

    def initialize(self, message, time, confidence):
        result = {k: message[k] for k in ('boxes', 'scores', 'points', 'features', 'valid')}
        result.update(hidden=self.encode(message, confidence),
                      velocity=message['boxes'].new_zeros(len(message['boxes']), 2),
                      last_seen=message['boxes'].new_full((len(message['boxes']),), time),
                      confidence=message['boxes'].new_full((len(message['boxes']), 1), confidence))
        return result

    def propagate(self, state, dt):
        boxes = state['boxes']
        elapsed = boxes.new_full((len(boxes), 1), max(0., dt))
        motion = self.motion(torch.cat((state['hidden'], state['velocity']/20,
                                       elapsed, state['confidence']), 1)).tanh()
        center = boxes[:, :2] + state['velocity']*elapsed + .5*self.settings['max_acceleration']*motion[:, :2]*elapsed.square()
        new_boxes = torch.cat((center, boxes[:, 2:6],
                               boxes[:, 6:7] + motion[:, 2:3]*elapsed), 1)
        new_state = dict(state)
        new_state.update(boxes=new_boxes, points=move_keypoints(state['points'], boxes, new_boxes))
        return new_state

    def sequence(self, messages, times, confidences):
        state = self.initialize(messages[-1], times[-1], confidences[-1])
        losses, counts = [], {'matched': 0, 'births': len(state['boxes']), 'retained_missing': 0}
        previous = times[-1]
        for index in reversed(range(len(messages)-1)):
            time = times[index]
            pred = self.propagate(state, time-previous)
            obs = messages[index]
            old_ids, new_ids = associate(pred['boxes'], obs['boxes'], self.settings['match_radius'])
            updated = self.initialize(obs, time, confidences[index])
            if len(old_ids):
                if time-previous > 1e-6:
                    losses.append(F.smooth_l1_loss(pred['boxes'][old_ids, :2], obs['boxes'][new_ids, :2].detach()))
                hidden = updated['hidden'].clone()
                hidden[new_ids] = self.recurrent(updated['hidden'][new_ids], pred['hidden'][old_ids])
                velocity = updated['velocity'].clone()
                if time-previous > 1e-6:
                    velocity[new_ids] = ((obs['boxes'][new_ids, :2]-state['boxes'][old_ids, :2])/(time-previous)).clamp(-35, 35)
                else:
                    velocity[new_ids] = state['velocity'][old_ids]
                updated.update(hidden=hidden, velocity=velocity)
            missing = torch.ones(len(pred['boxes']), dtype=torch.bool, device=pred['boxes'].device)
            missing[old_ids] = False
            missing &= (time-pred['last_seen']) <= self.settings['max_missing_age_s']
            counts['matched'] += len(old_ids)
            counts['births'] += len(obs['boxes'])-len(new_ids)
            counts['retained_missing'] += int(missing.sum())
            for key in updated:
                updated[key] = torch.cat((updated[key], pred[key][missing]), 0)
            state, previous = updated, time
        state = self.propagate(state, -previous)
        keep = (0-state['last_seen']) <= self.settings['max_missing_age_s']
        state = {k: v[keep] for k, v in state.items()}
        zero = next(self.parameters()).sum()*0
        return state, torch.stack(losses).mean() if losses else zero, counts

    @staticmethod
    def transform_message(message, transform):
        transform = transform.to(message['boxes'])
        out = dict(message)
        out['boxes'] = transform_boxes(message['boxes'], transform)
        out['points'] = transform_points(message['points'], transform)
        out['background'] = dict(message['background'])
        out['background']['xyz'] = transform_points(message['background']['xyz'], transform)
        return out

    def forward(self, messages, transforms, times, ego_histories, history_transforms, history_valid):
        # One scene per batch, arbitrary collaborator count except DAIR history adapter.
        k = times.shape[1]
        ego = self.transform_message(messages[0], transforms[0, 0])
        states = [self.initialize(ego, 0., 1.)]
        motion_losses, registrations = [], []
        counts = {'matched': 0, 'births': 0, 'retained_missing': 0}
        for cav in range(1, len(times)):
            sequence, confidence = [], []
            for frame in range(k):
                msg = self.transform_message(messages[cav*k+frame], transforms[cav, frame])
                if bool(history_valid[frame]):
                    reference = self.transform_message(ego_histories[frame], history_transforms[frame])
                    correction, stats = register_background(msg['background'], reference['background'])
                    msg = self.transform_message(msg, correction)
                else:
                    stats = {'accepted': 0., 'matches': 0., 'residual': 0., 'confidence': 0.}
                registrations.append(stats)
                sequence.append(msg)
                confidence.append(stats['confidence'])
            state, loss, scene_counts = self.sequence(sequence, times[cav].tolist(), confidence)
            states.append(state)
            motion_losses.append(loss)
            for key in counts:
                counts[key] += scene_counts[key]
        all_state = {key: torch.cat([s[key] for s in states], 0) for key in states[0]}
        boxes = all_state['boxes']
        n = len(boxes)
        if n == 0:
            return {'boxes': boxes, 'logits': boxes.new_zeros(0),
                    'motion_loss': next(self.parameters()).sum()*0, 'diagnostics': counts}
        xyz = all_state['points'].reshape(-1, 3)
        features = all_state['features'].reshape(-1, all_state['features'].shape[-1])
        valid = all_state['valid'].reshape(-1)
        xyz, features = xyz[valid], features[valid]
        contexts = []
        # Chunk queries to bound temporary N x K x C memory.
        for start in range(0, n, 16):
            qbox = boxes[start:start+16]
            h = all_state['hidden'][start:start+16]
            if len(xyz) == 0:
                contexts.append(torch.zeros_like(h))
                continue
            relative = xyz[None] - qbox[:, None, :3]
            near = relative[..., :2].norm(dim=-1) < self.settings['fusion_radius']
            point = self.point_encoder(torch.cat((features[None].expand(len(h), -1, -1), relative/10), -1))
            attention = (point*self.query(h)[:, None]).sum(-1)/(h.shape[-1]**.5)
            weights = attention.masked_fill(~near, -1e4).softmax(-1)*near
            weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-6)
            contexts.append((point*weights[..., None]).sum(1))
        delta = self.refiner(torch.cat((all_state['hidden'], torch.cat(contexts)), 1))
        refined = torch.cat((boxes[:, :3]+delta[:, :3].tanh()*2,
                              boxes[:, 3:6]*torch.exp(delta[:, 3:6].tanh()*.3),
                              boxes[:, 6:7]+delta[:, 6:7].tanh()*.5), 1)
        logits = torch.logit(all_state['scores'].clamp(.01, .99)) + delta[:, 7]
        for key in ('accepted', 'matches', 'residual', 'confidence'):
            counts['background_'+key] = sum(r[key] for r in registrations)/max(1, len(registrations))
        counts['instances'] = n
        counts['keypoints'] = int(valid.sum())
        return {'boxes': refined, 'logits': logits,
                'motion_loss': torch.stack(motion_losses).mean() if motion_losses else delta.sum()*0,
                'diagnostics': counts}


def capri_loss(output, gt):
    boxes, logits = output['boxes'], output['logits']
    gt = gt.to(boxes)
    pred_ids, gt_ids = associate(boxes, gt, radius=4.)
    labels = logits.new_zeros(logits.shape)
    labels[pred_ids] = 1
    zero = output['motion_loss']*0
    classification = F.binary_cross_entropy_with_logits(logits, labels) if len(logits) else zero
    regression = zero
    if len(pred_ids):
        a, b = boxes[pred_ids], gt[gt_ids]
        regression = (F.smooth_l1_loss(a[:, :3], b[:, :3]) +
                      F.smooth_l1_loss(a[:, 3:6].log(), b[:, 3:6].clamp_min(.1).log()) +
                      (1-torch.cos(a[:, 6]-b[:, 6])).mean())
    loss = classification+2*regression+.1*output['motion_loss']
    return loss, {'classification': float(classification.detach()), 'regression': float(regression.detach()),
                  'motion': float(output['motion_loss'].detach()), 'matched_gt': len(gt_ids), 'gt_count': len(gt)}
