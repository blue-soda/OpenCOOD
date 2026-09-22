"""Independent per-instance recurrent motion; immutable observation caches."""
from dataclasses import dataclass, replace
import torch
from torch import nn
from .geometry import associate, transport


@dataclass
class Track:
    box: torch.Tensor
    hidden: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    yaw_rate: torch.Tensor
    time: float
    observed_box: torch.Tensor
    observed_time: float
    score: torch.Tensor
    caches: dict
    observations: int = 1
    observed_source: str = ''


class Temporal(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        h = settings['hidden_dim']
        # Geometry-only input decouples temporal encoding from D's feature codec.
        self.encoder = nn.Sequential(nn.Linear(17, h), nn.LayerNorm(h), nn.ReLU())
        self.recurrent = nn.GRUCell(h, h)
        self.motion = nn.Sequential(nn.Linear(h+2, h), nn.ReLU(), nn.Linear(h, 3))
        nn.init.zeros_(self.motion[-1].weight)
        nn.init.zeros_(self.motion[-1].bias)

    def observation_hidden(self, box, score, old=None, dt=0., quality=1.):
        velocity = box.new_zeros(2) if old is None else old.velocity
        residual = box.new_zeros(3) if old is None else torch.cat((box[:2]-old.box[:2], torch.sin(box[6:7]-old.box[6:7])))
        vector = torch.cat((box[:3]/100, box[3:6]/10, box[6:7].sin(), box[6:7].cos(),
                            score.reshape(1), velocity/30, residual/5, box.new_tensor([dt, quality, float(old is not None)])))
        encoded = self.encoder(vector)
        hidden = encoded.new_zeros(encoded.shape) if old is None else old.hidden
        return self.recurrent(encoded[None], hidden[None])[0]

    def predict(self, track, time, mode='learned'):
        dt = time-track.time
        if dt < -1e-7:
            raise ValueError('A past observation cannot update a future state')
        dt = max(0., dt)
        v = track.velocity if mode != 'none' else torch.zeros_like(track.velocity)
        a = track.acceleration if mode == 'learned' else torch.zeros_like(track.acceleration)
        omega = track.yaw_rate if mode != 'none' else torch.zeros_like(track.yaw_rate)
        box = torch.cat((track.box[:2]+v*dt+.5*a*dt*dt, track.box[2:6], track.box[6:7]+omega*dt))
        return replace(track, box=box, velocity=v+a*dt, time=time)

    def observe(self, message, index, time, old=None, mode='learned'):
        box, score = message['boxes'][index], message['scores'][index]
        source = str(message['metadata']['agent_id'])
        dt = 0. if old is None else time-old.observed_time
        if old is not None and self.settings.get('axial_observations', False):
            # Frozen detector headings have a pi ambiguity. Resolve the nearest
            # equivalent axis before estimating yaw rate or transporting caches.
            angle = box[6:7]-old.box[6:7]
            yaw = old.box[6:7]+.5*torch.atan2((2*angle).sin(), (2*angle).cos())
            box = torch.cat((box[:6], yaw))
        hidden = self.observation_hidden(box, score, old, dt, message.get('quality', 1.))
        velocity = box.new_zeros(2) if old is None else old.velocity
        omega = box.new_zeros(1) if old is None else old.yaw_rate
        if old is not None and dt > 1e-6:
            # Always last REAL observation, never the propagated box.
            velocity = ((box[:2]-old.observed_box[:2])/dt).clamp(-self.settings['max_speed'], self.settings['max_speed'])
            angle = box[6:7]-old.observed_box[6:7]
            omega = (torch.atan2(angle.sin(), angle.cos())/dt).clamp(-self.settings['max_yaw_rate'], self.settings['max_yaw_rate'])
        correction = self.motion(torch.cat((hidden, velocity/30))).tanh()
        observation_count = 1 if old is None else old.observations+1
        if observation_count < self.settings.get('minimum_motion_observations', 1):
            correction = correction*0
        acceleration = correction[:2]*self.settings['max_acceleration']
        if mode == 'learned':
            omega = (omega+correction[2:3]*self.settings['max_yaw_rate']).clamp(-self.settings['max_yaw_rate'], self.settings['max_yaw_rate'])
        caches = {} if old is None else dict(old.caches)
        ids = message['instance_ids'] == index
        # Keep each source's last nonempty cache. No repeated historical evidence.
        if bool(ids.any()):
            transform = message.get('nominal_to_reference')
            source_yaw = box.new_zeros(()) if transform is None else torch.atan2(transform[1, 0], transform[0, 0])
            caches[source] = dict(box=box, points=message['points'][ids], features=message['features'][ids],
                                  time=time, frame_id=message['metadata']['frame_id'], score=score, source_yaw=source_yaw)
        return Track(box, hidden, velocity, acceleration, omega, time, box, time, score, caches,
                     observation_count, source)

    def update(self, tracks, message, time, mode='learned'):
        predicted = [self.predict(t, time, mode) for t in tracks]
        predicted = [t for t in predicted if time-t.observed_time <= self.settings['ttl_s']]
        boxes = torch.stack([t.box for t in predicted]) if predicted else message['boxes'][:0]
        elapsed = max([time-t.observed_time for t in predicted] or [0.])
        radius = self.settings['match_radius_m']+self.settings['gate_growth_mps']*elapsed
        source = str(message['metadata']['agent_id'])
        if predicted and self.settings.get('cross_source_radius_m') is not None:
            radius = boxes.new_tensor([min(radius, self.settings['cross_source_radius_m'])
                if t.observed_source and t.observed_source != source else radius for t in predicted])[:,None]
        rows, cols = associate(boxes, message['boxes'], radius)
        matches = dict(zip(cols.tolist(), rows.tolist()))
        used = set(rows.tolist())
        result = [self.observe(message, j, time, predicted[matches[j]] if j in matches else None, mode) for j in range(len(message['boxes']))]
        result += [track for j, track in enumerate(predicted) if j not in used]
        axial_flips = sum(int(torch.cos(message['boxes'][col,6]-predicted[row].box[6]) < 0)
                          for col,row in matches.items())
        cross_matches = [(col,row) for col,row in matches.items()
                         if predicted[row].observed_source and predicted[row].observed_source != source]
        cross_far = sum(int((message['boxes'][col,:2]-predicted[row].box[:2]).norm() > 2.)
                        for col,row in cross_matches)
        return result, {'matched': len(matches), 'births': len(message['boxes'])-len(matches), 'missing': len(predicted)-len(matches),
                        'axial_flip_matches': axial_flips, 'cross_source_matches': len(cross_matches),
                        'cross_source_matches_over_2m': cross_far}

    def sequence(self, messages, query_time=0., mode='learned'):
        """Sample-local reference frame, dedup; simultaneous messages use a fixed order."""
        unique = {}
        for message in messages:
            metadata = message['metadata']
            if metadata['time_s'] > query_time+1e-7 or metadata.get('arrival_s', metadata['time_s']) > query_time+1e-7:
                continue
            key = (str(metadata['agent_id']), str(metadata['frame_id']))
            if key in unique and unique[key]['metadata']['time_s'] != metadata['time_s']:
                raise ValueError('Duplicate frame has conflicting observation times')
            unique[key] = message
        # Ego last at a tied time, allowing its actual current geometry to correct matches.
        ordered = sorted(unique.values(), key=lambda m: (m['metadata']['time_s'], str(m['metadata']['agent_id']) == '0', str(m['metadata']['agent_id']), str(m['metadata']['frame_id'])))
        sequences = {m['metadata']['sequence_id'] for m in ordered}
        if len(sequences) > 1:
            raise ValueError('Cross-scene recurrence is forbidden')
        tracks, counts = [], {'matched': 0, 'births': 0, 'missing': 0, 'deduplicated': len(messages)-len(unique),
                             'axial_flip_matches': 0, 'cross_source_matches': 0, 'cross_source_matches_over_2m': 0}
        for message in ordered:
            tracks, update = self.update(tracks, message, message['metadata']['time_s'], mode)
            for key in update:
                counts[key] += update[key]
        return [self.predict(t, query_time, mode) for t in tracks if query_time-t.observed_time <= self.settings['ttl_s']], counts

    def collect(self, tracks, query_time=0., exclude_source='0', include_orientation=False):
        points, features, ages, scores, orientations = [], [], [], [], []
        for track in tracks:
            for source, cache in sorted(track.caches.items()):
                age = query_time-cache['time']
                if source == exclude_source or age > self.settings['ttl_s']:
                    continue
                n = len(cache['points'])
                points.append(transport(cache['points'], cache['box'][None].expand(n, -1), track.box[None].expand(n, -1)))
                features.append(cache['features'])
                ages.append(track.box.new_full((n,), age))
                scores.append(cache['score'].expand(n))
                if include_orientation:
                    angle = cache.get('source_yaw', track.box.new_zeros(()))+track.box[6]-cache['box'][6]
                    orientations.append(torch.stack((angle.sin(), angle.cos())).expand(n, 2))
        if not points:
            return None
        output = tuple(torch.cat(x) for x in (points, features, ages, scores))
        return output+(torch.cat(orientations),) if include_orientation else output


class ReplayWindow:
    """Bounded arrival-order replay; caller supplies one fixed scene reference frame."""
    def __init__(self, temporal, window_s=2., max_messages=32):
        self.temporal, self.window_s, self.max_messages = temporal, window_s, max_messages
        self.scene, self.messages, self.rejected = None, {}, 0

    def receive(self, message, now, mode='learned'):
        meta = message['metadata']
        if self.scene != meta['sequence_id']:
            self.scene, self.messages = meta['sequence_id'], {}
        if meta.get('arrival_s', meta['time_s']) > now or meta['time_s'] > now or meta['time_s'] < now-self.window_s:
            self.rejected += 1
        else:
            self.messages[(str(meta['agent_id']), str(meta['frame_id']))] = message
        self.messages = {k: v for k, v in self.messages.items() if v['metadata']['time_s'] >= now-self.window_s}
        ordered = sorted(self.messages.items(), key=lambda kv: (kv[1]['metadata']['time_s'], kv[0]))[-self.max_messages:]
        self.messages = dict(ordered)
        return self.temporal.sequence(list(self.messages.values()), now, mode)
