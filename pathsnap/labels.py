"""Derive per-observation reference arc from the supplied .route file.

The .route file is only the ordered list of arcs, with no per-GPS-row labels.
We align the GPS sequence to the route by a monotone dynamic programme
(non-decreasing route index, bounded forward jumps) minimising squared
point-to-arc distance. This is a *labelling* step for training/evaluation only;
it uses the reference route and is never run on the prediction path.
"""
import numpy as np


def seg_dist(P, A, B):
    """Distance from points P (n,2) to segments A-B (m,2) -> (n,m), plus u."""
    AB = B - A
    l2 = (AB ** 2).sum(1) + 1e-9
    u = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / l2[None], 0, 1)
    proj = A[None] + u[..., None] * AB[None]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(2)), u


def align_to_route(trip, max_jump=25, chunk=2000):
    """Return ref_arc (n,) aligned to the GPS sequence."""
    route = trip.route
    fr, to = trip.arcs[route, 0], trip.arcs[route, 1]
    A, B = trip.nodes_xy[fr], trip.nodes_xy[to]
    n, R = len(trip.xy), len(route)
    D = np.empty((n, R), np.float32)
    for s in range(0, n, chunk):
        D[s:s + chunk] = seg_dist(trip.xy[s:s + chunk], A, B)[0]
    cost = np.minimum(D, 100.0) ** 2           # robust to outliers
    jump_pen = 4.0
    INF = 1e18
    acc = np.full(R, INF); acc[: max(5, max_jump)] = cost[0, : max(5, max_jump)]
    back = np.zeros((n, R), np.int16)
    for t in range(1, n):
        best = acc.copy(); arg = np.zeros(R, np.int16)
        for k in range(1, max_jump + 1):
            cand = np.full(R, INF); cand[k:] = acc[:-k] + jump_pen * k
            m = cand < best
            best = np.where(m, cand, best); arg = np.where(m, k, arg)
        acc = best + cost[t]; back[t] = arg
    # end anywhere in last quarter of route (trip may end before route does)
    j = int(np.argmin(acc)); idx = np.empty(n, np.int64)
    for t in range(n - 1, -1, -1):
        idx[t] = j; j -= int(back[t, j])
    return route[idx]
