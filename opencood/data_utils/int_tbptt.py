"""TBPTT boundaries and delayed optimizer updates, independent of LiDAR IO."""
import torch


def window_ends(rows, length, max_gap_s=.25):
    if length < 1:
        raise ValueError('TBPTT length must be positive')
    count, ends = 0, set()
    for index, row in enumerate(rows):
        count += 1
        boundary = index == len(rows)-1
        if not boundary:
            following = rows[index+1]
            delta_t = (following['timestamp_us']-row['timestamp_us'])/1e6
            boundary = (following['segment'] != row['segment'] or
                        following.get('reset_before', False) or delta_t > max_gap_s)
            if not boundary and delta_t <= 0:
                raise ValueError('Noncausal TBPTT window')
        if boundary or count == length:
            ends.add(index)
            count = 0
    return ends


class WindowOptimizer:
    """Keep weights fixed while building a window, then detach but retain state."""
    def __init__(self, model, optimizer):
        self.model, self.optimizer, self.losses = model, optimizer, []
        optimizer.zero_grad(set_to_none=True)

    def add(self, loss):
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        self.losses.append(loss)

    def finish(self, state):
        result = dict(supervised_frames=len(self.losses), optimizer_step=False)
        if self.losses:
            torch.stack(self.losses).mean().backward()
            temporal = [p for p in self.model.feature_memory.parameters() if p.grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            if not torch.isfinite(norm):
                raise RuntimeError('Non-finite gradient')
            result.update(gradient_norm=float(norm), temporal_gradient_norm=sum(
                float(p.grad.detach().square().sum()) for p in temporal)**.5,
                optimizer_step=True)
            self.optimizer.step()
        self.losses.clear()
        self.optimizer.zero_grad(set_to_none=True)
        if state is not None:
            state = dict(state, value=state['value'].detach())
        return state, result
