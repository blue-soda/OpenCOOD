"""Audit causal persistence of corrections observed at executed middle cells."""
import torch

from opencood.models.sub_modules.int_feature_memory import warp_bev


class CorrectionTraceAudit:
    def __init__(self):
        self.by_scene = {}
        self.reset()

    def reset(self):
        self.memory = None
        self.pose = None

    def observe(self, meta, area, executed, valid, quiet, actual, prediction, teacher):
        # The last channel carries validity through exactly the same nearest-neighbor warp.
        aligned = (torch.zeros_like(torch.cat((actual, valid.float()), dim=1))
                   if self.memory is None else warp_bev(self.memory, self.pose, meta['pose'], area)[0])
        sites = quiet.flatten().nonzero().flatten()[::16]
        visible = aligned[0, -1].flatten()[sites].bool()
        error = (teacher - prediction)[0].flatten(1)[:, sites].double()
        target = error[:, visible]
        x = aligned[0, :-1].flatten(1)[:, sites[visible]].double()
        sign = x.sign()
        shuffled = sign.roll(sign.shape[1] // 2, dims=1) if sign.numel() else sign
        stats = dict(total_y2=error.square().sum(1), y2=target.square().sum(1), y=target.sum(1))
        for name, value in [('amplitude', x), ('sign', sign), ('shuffled', shuffled)]:
            stats[name+'_x'] = value.sum(1)
            stats[name+'_xx'] = value.square().sum(1)
            stats[name+'_xy'] = (value * target).sum(1)
        for key, value in [('samples', len(sites)), ('covered', target.shape[1])]:
            stats[key] = torch.tensor(value, device=actual.device)
        scene = meta['scene']
        if scene in self.by_scene:
            for key, value in stats.items():
                self.by_scene[scene][key] += value
        else:
            self.by_scene[scene] = stats

        # Never write teacher values. Only previously observable errors can enter memory.
        observed = executed & valid.bool()
        self.memory = torch.cat((torch.where(observed, actual - prediction, aligned[:, :-1]),
                                 (observed | aligned[:, -1:].bool()).float()), dim=1)
        self.pose = meta['pose']

    def report(self):
        return {scene: {key: value.cpu().tolist() for key, value in sums.items()}
                for scene, sums in self.by_scene.items()}
