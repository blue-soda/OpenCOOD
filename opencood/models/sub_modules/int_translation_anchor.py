"""Five-scan translation anchor snapped to the detection lattice, without interpolation."""
import math

import torch.nn.functional as F


class TranslationAnchor:
    stride = 2
    margin = 16
    period = 5

    def __init__(self, voxel_xy):
        self.voxel_xy = voxel_xy
        self.pose = None
        self.age = 0

    def locate(self, pose, boundary):
        reset = boundary or self.pose is None or self.age >= self.period
        x = y = 0
        if not reset:
            dx = pose[0][3] - self.pose[0][3]
            dy = pose[1][3] - self.pose[1][3]
            yaw = math.atan2(self.pose[1][0], self.pose[0][0])
            c,s = math.cos(yaw),math.sin(yaw)
            x = self.stride * round((c*dx+s*dy)/(self.stride*self.voxel_xy[0]))
            y = self.stride * round((-s*dx+c*dy)/(self.stride*self.voxel_xy[1]))
            reset = max(abs(x),abs(y)) > self.margin
        if reset:
            self.pose, self.age, x, y = pose, 0, 0, 0
        self.age += 1
        return reset, (x,y)

    @classmethod
    def place(cls, spatial, shift):
        x,y = shift
        h,w = spatial.shape[-2:]
        left,top = cls.margin+x,cls.margin+y
        canvas = F.pad(spatial,(left,cls.margin-x,top,cls.margin-y))
        assert all(v%cls.stride==0 for v in (left,top,h,w))
        crop = (slice(top//2,(top+h)//2),slice(left//2,(left+w)//2))
        return canvas,crop
