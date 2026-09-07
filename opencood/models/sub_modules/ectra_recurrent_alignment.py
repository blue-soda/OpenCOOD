import torch
import torch.nn as nn
import torch.nn.functional as F


class EctraRecurrentAlignment(nn.Module):
    """
    Ego-Calibrated Temporal Recurrent Alignment.

    This first runnable version keeps the state as dense BEV features. It uses a
    continuous delta-t map to predict hidden-state motion, observation residual
    calibration, and prediction/observation trust gates.
    """

    def __init__(self, args):
        super(EctraRecurrentAlignment, self).__init__()
        self.feature_dim = args.get('feature_dim', 64)
        self.time_scale = float(args.get('time_scale', 10.0))
        self.motion_range = float(args.get('motion_range', 8.0))
        self.calib_range = float(args.get('calib_range', 4.0))
        hidden_dim = int(args.get('hidden_dim', self.feature_dim))
        gate_dim = int(args.get('gate_dim', max(16, self.feature_dim // 2)))

        motion_in = self.feature_dim * 2 + 2
        self.motion_net = nn.Sequential(
            nn.Conv2d(motion_in, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)
        )

        calib_in = self.feature_dim * 3 + 2
        self.calib_net = nn.Sequential(
            nn.Conv2d(calib_in, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        )

        trust_in = self.feature_dim * 3 + 4
        self.trust_net = nn.Sequential(
            nn.Conv2d(trust_in, gate_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(gate_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_dim, 3, kernel_size=3, padding=1)
        )

        cand_in = self.feature_dim * 3 + 2
        self.candidate_net = nn.Sequential(
            nn.Conv2d(cand_in, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )

    @staticmethod
    def _warp_by_flow(x, flow):
        n, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(n, 1, 1, 1)
        flow_x = flow[:, 0] * (2.0 / max(w - 1, 1))
        flow_y = flow[:, 1] * (2.0 / max(h - 1, 1))
        grid = base_grid + torch.stack((flow_x, flow_y), dim=-1)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros',
                             align_corners=True)

    def _time_maps(self, dt, like):
        n, _, h, w = like.shape
        dt = dt.to(device=like.device, dtype=like.dtype).view(n, 1, 1, 1)
        norm_dt = torch.clamp(dt / self.time_scale, min=0.0, max=1.0)
        return torch.cat((
            norm_dt.expand(n, 1, h, w),
            torch.log1p(torch.clamp(dt, min=0.0)).expand(n, 1, h, w)
        ), dim=1)

    def _propagate(self, hidden, ego_ref, dt):
        time_maps = self._time_maps(dt, hidden)
        motion = self.motion_net(torch.cat((hidden, ego_ref, time_maps), dim=1))
        flow = torch.tanh(motion[:, :2]) * self.motion_range
        gamma = torch.sigmoid(motion[:, 2:3])
        hidden_pred = gamma * self._warp_by_flow(hidden, flow)
        return hidden_pred, flow, gamma

    def _update(self, hidden_pred, obs, ego_ref, dt):
        time_maps = self._time_maps(dt, hidden_pred)
        calib_flow = torch.tanh(
            self.calib_net(torch.cat((obs, ego_ref, hidden_pred, time_maps), dim=1))
        ) * self.calib_range
        obs_calib = self._warp_by_flow(obs, calib_flow)

        pred_residual = torch.mean(torch.abs(hidden_pred - obs_calib), dim=1,
                                   keepdim=True)
        obs_residual = torch.mean(torch.abs(obs_calib - ego_ref), dim=1,
                                  keepdim=True)
        trust_logits = self.trust_net(torch.cat((
            hidden_pred, obs_calib, ego_ref, pred_residual, obs_residual, time_maps
        ), dim=1))
        r_pred = torch.sigmoid(trust_logits[:, 0:1])
        r_obs = torch.sigmoid(trust_logits[:, 1:2])
        z = torch.sigmoid(trust_logits[:, 2:3])

        candidate = self.candidate_net(torch.cat((
            r_pred * hidden_pred, r_obs * obs_calib, ego_ref, r_pred, r_obs
        ), dim=1))
        write = z * r_obs
        hidden = (1.0 - write) * hidden_pred + write * candidate
        trust = torch.clamp(0.5 * (r_pred + r_obs), 0.0, 1.0)
        return hidden, obs_calib, calib_flow, trust, r_pred, r_obs

    def _reshape_time_intervals(self, time_intervals, record_len, k, device):
        if time_intervals is None:
            total_cav = int(torch.sum(record_len).item())
            return torch.zeros(total_cav, k, device=device)
        if not torch.is_tensor(time_intervals):
            time_intervals = torch.as_tensor(time_intervals)
        time_intervals = time_intervals.to(device=device, dtype=torch.float32).view(-1)
        expected = int(torch.sum(record_len).item()) * k
        if time_intervals.numel() < expected:
            pad = torch.zeros(expected - time_intervals.numel(), device=device)
            time_intervals = torch.cat((time_intervals, pad), dim=0)
        return time_intervals[:expected].view(-1, k)

    def forward(self, features, record_len, time_intervals=None):
        if features.numel() == 0:
            return features, {}

        total_frames, c, h, w = features.shape
        total_cav = int(torch.sum(record_len).item())
        if total_cav == 0 or total_frames % total_cav != 0:
            return features, {}

        k = total_frames // total_cav
        if k <= 1:
            return features, {}

        intervals = self._reshape_time_intervals(
            time_intervals, record_len, k, features.device)
        chunks = torch.tensor_split(
            features, torch.cumsum(record_len * k, dim=0)[:-1].cpu())

        aligned_chunks = []
        trust_maps = []
        motion_losses = []
        cav_offset = 0
        for batch_idx, batch_features in enumerate(chunks):
            cav_num = int(record_len[batch_idx].item())
            nodes = batch_features.view(cav_num, k, c, h, w)
            ego_seq = nodes[0]
            batch_intervals = intervals[cav_offset:cav_offset + cav_num]
            cav_offset += cav_num

            cav_sequences = [nodes[0]]
            for cav_idx in range(1, cav_num):
                hidden = nodes[cav_idx, k - 1:k]
                prev_time = torch.abs(batch_intervals[cav_idx, k - 1:k])
                last_trust = torch.ones(1, 1, h, w, device=features.device,
                                        dtype=features.dtype)

                for frame_idx in range(k - 2, -1, -1):
                    curr_time = torch.abs(batch_intervals[cav_idx, frame_idx:frame_idx + 1])
                    dt = torch.clamp(prev_time - curr_time, min=0.0)
                    ego_ref = ego_seq[frame_idx:frame_idx + 1]
                    hidden_pred, _, _ = self._propagate(hidden, ego_ref, dt)
                    hidden, obs_calib, _, last_trust, _, _ = self._update(
                        hidden_pred, nodes[cav_idx, frame_idx:frame_idx + 1],
                        ego_ref, dt)
                    motion_losses.append(
                        F.smooth_l1_loss(hidden_pred, obs_calib.detach(),
                                         reduction='mean'))
                    prev_time = curr_time

                final_dt = torch.abs(batch_intervals[cav_idx, 0:1])
                if torch.any(final_dt > 0):
                    hidden, _, _ = self._propagate(hidden, ego_seq[0:1], final_dt)

                updated_current = last_trust * hidden + \
                    (1.0 - last_trust) * nodes[cav_idx, 0:1]
                cav_sequences.append(torch.cat(
                    (updated_current, nodes[cav_idx, 1:]), dim=0))
                trust_maps.append(last_trust)

            aligned_chunks.append(torch.stack(cav_sequences, dim=0).view(cav_num * k, c, h, w))

        aligned = torch.cat(aligned_chunks, dim=0)
        aux = {}
        if trust_maps:
            aux['ectra_trust'] = torch.cat(trust_maps, dim=0)
        if motion_losses:
            aux['ectra_motion_loss'] = torch.stack(motion_losses).mean()
        return aligned, aux
