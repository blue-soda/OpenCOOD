"""Current sensor samples and detection-grid queries in a five-scan reference frame."""
import math

from torch.nn import functional as F


def attach_window(rows, rigid):
    placed = []
    for i, row in enumerate(rows):
        first = i == 0 or row.get('reset_before') or row['segment'] != rows[i-1]['segment']
        pose = row['pose']
        yaw = math.atan2(pose[1][0], pose[0][0])
        if first:
            origin = (pose[0][3],pose[1][3])
            heading = yaw
        transform = None
        if rigid and not first:
            dx, dy = pose[0][3]-origin[0], pose[1][3]-origin[1]
            c,s = math.cos(heading),math.sin(heading)
            transform = (math.cos(yaw-heading),math.sin(yaw-heading),c*dx+s*dy,-s*dx+c*dy)
            if transform == (1.,0.,0.,0.):
                transform = None
        placed.append(dict(row, window_transform=transform, window_ordinary=not first))
    return placed


def place_points(points, transform):
    c,s,dx,dy = transform
    result = points.copy()
    result[:,0] = c*points[:,0]-s*points[:,1]+dx
    result[:,1] = s*points[:,0]+c*points[:,1]+dy
    return result


def current_readout(features, transform, bounds):
    """Query each current cell center in the reference map using metric coordinates."""
    c,s,dx,dy = transform
    xmin,ymin,_,xmax,ymax,_ = bounds
    hx,hy = (xmax-xmin)/2,(ymax-ymin)/2
    cx,cy = xmin+hx,ymin+hy
    theta = features.new_tensor([[[c,-s*hy/hx,(c*cx-s*cy+dx-cx)/hx],
                                  [s*hx/hy,c,(s*cx+c*cy+dy-cy)/hy]]])
    grid = F.affine_grid(theta,features.shape,align_corners=False)
    return F.grid_sample(features,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
