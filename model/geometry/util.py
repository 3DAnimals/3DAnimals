import torch
from einops import repeat


def sample_farthest_points(pts, k, return_index=False):
    b, c, n = pts.shape
    farthest_pts = torch.zeros((b, 3, k), device=pts.device, dtype=pts.dtype)
    indexes = torch.zeros((b, k), device=pts.device, dtype=torch.int64)
    
    index = torch.randint(n, [b], device=pts.device)
    
    gather_index = repeat(index, 'b -> b c 1', c=c)
    farthest_pts[:, :, 0] = torch.gather(pts, 2, gather_index)[:, :, 0]
    indexes[:, 0] = index
    distances = torch.norm(farthest_pts[:, :, 0][:, :, None] - pts, dim=1)
    
    for i in range(1, k):
        _, index = torch.max(distances, dim=1)
        gather_index = repeat(index, 'b -> b c 1', c=c)
        farthest_pts[:, :, i] = torch.gather(pts, 2, gather_index)[:, :, 0]
        indexes[:, i] = index
        distances = torch.min(distances, torch.norm(farthest_pts[:, :, i][:, :, None] - pts, dim=1))

    if return_index:
        return farthest_pts, indexes
    else:
        return farthest_pts


def line_segment_distance(a, b, points, sqrt=True):
    """
    compute the distance between a point and a line segment defined by a and b
    a, b: ... x D
    points: ... x D
    """
    def sumprod(x, y, keepdim=True):
        return torch.sum(x * y, dim=-1, keepdim=keepdim)

    a, b = a[..., None, :], b[..., None, :]

    t_min = sumprod(points - a, b - a) / torch.max(sumprod(b - a, b - a), torch.tensor(1e-6, device=a.device))
    
    t_line = torch.clamp(t_min, 0.0, 1.0)

    # closest points on the line to every point
    s = a + t_line * (b - a)

    distance = sumprod(s - points, s - points, keepdim=False)
    
    if sqrt:
        distance = torch.sqrt(distance + 1e-6)

    return distance
