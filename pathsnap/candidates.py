"""Candidate road-segment generation and per-(observation, candidate) features."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .labels import seg_dist

RADIUS = 80.0     # metres: candidate search radius
K_MAX = 12        # max candidates per observation
DENSIFY = 15.0    # metres between sample points along arcs (for KD-tree)

FEATURES = ["d", "d_rel", "d_ratio", "rank", "n_cand", "frac", "arc_len",
            "d_smooth", "d_smooth_rel", "cos_head", "cos_head_wide", "speed",
            "dt_prev", "supp_prev", "supp_next", "out_deg", "in_deg",
            "same_prev_cand", "same_next_cand"]


def _arc_geometry(trip):
    A = trip.nodes_xy[trip.arcs[:, 0]]
    B = trip.nodes_xy[trip.arcs[:, 1]]
    return A, B


def generate(trip, radius=RADIUS, k_max=K_MAX):
    """Return DataFrame-like dict of arrays: obs, arc, d, u (projection frac)."""
    A, B = _arc_geometry(trip)
    P = trip.xy
    lo, hi = P.min(0) - radius - 5, P.max(0) + radius + 5
    inb = ((np.minimum(A, B) <= hi).all(1)) & ((np.maximum(A, B) >= lo).all(1))
    arc_ids = np.nonzero(inb)[0]
    a, b = A[arc_ids], B[arc_ids]
    L = np.hypot(*(b - a).T)
    ns = np.maximum(1, np.ceil(L / DENSIFY).astype(int)) + 1
    arc_rep = np.repeat(np.arange(len(arc_ids)), ns)
    off = np.concatenate([[0], np.cumsum(ns)[:-1]])
    pos = np.arange(ns.sum()) - np.repeat(off, ns)
    f = pos / np.repeat(ns - 1, ns)
    pts = a[arc_rep] + f[:, None] * (b - a)[arc_rep]
    tree = cKDTree(pts)
    n = len(P)
    kq = 48
    dist, idx = tree.query(P, k=kq, distance_upper_bound=radius)
    valid = np.isfinite(dist)
    obs = np.repeat(np.arange(n), kq)[valid.ravel()]
    loc = arc_rep[idx.ravel()[valid.ravel()]]
    key = np.unique(obs.astype(np.int64) * (len(arc_ids) + 1) + loc)
    obs, loc = key // (len(arc_ids) + 1), key % (len(arc_ids) + 1)
    arc = arc_ids[loc]
    return _finish(trip, obs, arc, A, B, k_max)


def _finish(trip, obs, arc, A, B, k_max):
    P = trip.xy
    ab = B[arc] - A[arc]
    l2 = (ab ** 2).sum(1) + 1e-9
    u = np.clip(((P[obs] - A[arc]) * ab).sum(1) / l2, 0, 1)
    d = np.hypot(*(P[obs] - (A[arc] + u[:, None] * ab)).T)
    order = np.lexsort((d, obs))
    obs, arc, d, u = obs[order], arc[order], d[order], u[order]
    # rank within observation, keep top k_max
    first = np.r_[0, np.nonzero(np.diff(obs))[0] + 1]
    cnt = np.diff(np.r_[first, len(obs)])
    rank = np.arange(len(obs)) - np.repeat(first, cnt)
    keep = rank < k_max
    return dict(obs=obs[keep], arc=arc[keep], d=d[keep], u=u[keep], rank=rank[keep])


def _bearing(P, w):
    n = len(P)
    i0 = np.clip(np.arange(n) - w, 0, n - 1); i1 = np.clip(np.arange(n) + w, 0, n - 1)
    v = P[i1] - P[i0]
    dt = np.maximum(i1 - i0, 1)
    return v, np.hypot(*v.T) / dt


def features(trip, cand):
    """Build the feature matrix (dict of arrays, columns FEATURES)."""
    P = trip.xy; n = len(P)
    A, B = _arc_geometry(trip)
    obs, arc = cand["obs"], cand["arc"]
    d, u, rank = cand["d"], cand["u"], cand["rank"]
    m = len(obs)
    first = np.r_[0, np.nonzero(np.diff(obs))[0] + 1]
    cnt = np.diff(np.r_[first, m])
    n_cand = np.zeros(n); n_cand[obs[first]] = cnt
    dmin = np.zeros(n) + 1e3; dmin[obs[first]] = d[first]
    ab = B[arc] - A[arc]; L = np.hypot(*ab.T) + 1e-9
    # heading agreement: arc direction vs smoothed motion direction
    v1, sp1 = _bearing(P, 2); v2, _ = _bearing(P, 6)
    def cosh(v):
        nv = np.hypot(*v.T) + 1e-9
        c = (ab * v[obs]).sum(1) / (L * nv[obs])
        return np.where(nv[obs] < 3.0, 0.0, c)   # undefined when ~stationary
    # smoothed distance of neighbouring GPS points (t-2..t+2) to the same arc
    ds = np.zeros(m)
    for o in (-2, -1, 1, 2):
        j = np.clip(obs + o, 0, n - 1)
        uu = np.clip(((P[j] - A[arc]) * ab).sum(1) / (L ** 2), 0, 1)
        ds += np.hypot(*(P[j] - (A[arc] + uu[:, None] * ab)).T)
    ds = (ds + d) / 5.0
    # smoothed distance relative to best smoothed distance in the observation
    dsmin = np.full(n, 1e9); np.minimum.at(dsmin, obs, ds)
    # network-support features: does this arc connect to a candidate at t-1 / t+1?
    na = len(trip.arcs)
    frm, to = trip.arcs[arc, 0], trip.arcs[arc, 1]
    by_obs_to = {}; by_obs_from = {}; by_obs_arc = {}
    for t0, t1 in zip(first, np.r_[first[1:], m]):
        o = obs[t0]
        by_obs_to[o] = set(to[t0:t1].tolist()); by_obs_from[o] = set(frm[t0:t1].tolist())
        by_obs_arc[o] = set(arc[t0:t1].tolist())
    supp_prev = np.zeros(m); supp_next = np.zeros(m)
    same_p = np.zeros(m); same_n = np.zeros(m)
    for i in range(m):
        o = obs[i]
        pv = by_obs_to.get(o - 1); nx = by_obs_from.get(o + 1)
        if pv is not None: supp_prev[i] = float(frm[i] in pv)
        if nx is not None: supp_next[i] = float(to[i] in nx)
        pa = by_obs_arc.get(o - 1); na_ = by_obs_arc.get(o + 1)
        if pa is not None: same_p[i] = float(arc[i] in pa)
        if na_ is not None: same_n[i] = float(arc[i] in na_)
    outdeg = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes_xy))
    indeg = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes_xy))
    dtp = np.r_[1.0, np.diff(trip.t)]
    F = dict(
        d=d, d_rel=d - dmin[obs], d_ratio=d / (dmin[obs] + 1.0), rank=rank.astype(float),
        n_cand=n_cand[obs], frac=u, arc_len=L, d_smooth=ds, d_smooth_rel=ds - dsmin[obs],
        cos_head=cosh(v1), cos_head_wide=cosh(v2), speed=sp1[obs], dt_prev=dtp[obs],
        supp_prev=supp_prev, supp_next=supp_next,
        out_deg=outdeg[to].astype(float), in_deg=indeg[frm].astype(float),
        same_prev_cand=same_p, same_next_cand=same_n)
    return np.column_stack([F[k] for k in FEATURES]).astype(np.float32)
