"""HMM / Viterbi route reconstruction on top of classifier candidate scores.

Hidden state   : at observation t, one of the K_t candidate road segments (arc ids) generated
                 for that GPS fix (candidate -> arc correspondence is kept in `arcs[t]`).
Emission       : log P(obs | arc) = alpha * log( p_clf(arc) / sum_k p_clf(k) )   -- the classifier's
                 candidate probability, renormalised per observation, tempered by alpha.
Transition     : Newson-Krumm style. With d_gps = |x_t - x_{t-1}| and d_net the along-network
                 distance between the projected positions on the two candidate arcs,
                     log P(b|a) = -|d_net - d_gps| / beta
                 Same-arc moves use the fraction difference; moves to a successor arc go via the
                 remaining length of a, a bounded Dijkstra between arc end/start nodes and the offset
                 into b. Unreachable (within the cutoff) pairs get a flat log-penalty `invalid_pen`
                 (so a decoding always exists: "invalid transitions" are allowed but very expensive),
                 immediate U-turns (b is the reverse of a) get an extra `uturn_pen`.
GPS gaps       : if the time gap exceeds `gap_s`, the chain is re-started (uniform transition),
                 because distance-based reachability is meaningless across long outages.
No candidates  : the observation is skipped by the chain and assigned the previously decoded arc
                 (or the next one, at the start of the trip).
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class HMMParams:
    alpha: float = 1.0
    beta: float = 30.0
    invalid_pen: float = 12.0
    uturn_pen: float = 4.0
    gap_s: float = 60.0
    cutoff_mult: float = 3.0
    cutoff_min: float = 250.0
    cutoff_max: float = 4000.0
    back_tol: float = 0.2   # tolerated backward move (fraction of arc) on the same arc

    def to_dict(self): return asdict(self)


class Net:
    """Directed road graph with bounded Dijkstra (cached)."""
    def __init__(self, arcs, nodes_xy):
        self.arcs = arcs
        a, b = nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]
        self.len = np.hypot(*(b - a).T)
        order = np.argsort(arcs[:, 0], kind="stable")
        self.ptr = np.zeros(len(nodes_xy) + 1, np.int64)
        np.add.at(self.ptr, arcs[:, 0] + 1, 1); self.ptr = np.cumsum(self.ptr)
        self.out_arc = order
        self.cache = {}

    def dijkstra(self, src, cutoff, with_pred=False):
        key = (src, int(np.ceil(cutoff / 100.0)), with_pred)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        lim = float(np.ceil(cutoff / 100.0) * 100.0)
        dist = {src: 0.0}; pred = {}; pq = [(0.0, src)]; done = set()
        ptr, oa, arcs, ln = self.ptr, self.out_arc, self.arcs, self.len
        while pq:
            d, u = heapq.heappop(pq)
            if u in done: continue
            done.add(u)
            for e in oa[ptr[u]:ptr[u + 1]]:
                v = int(arcs[e, 1]); nd = d + ln[e]
                if nd <= lim and nd < dist.get(v, 1e18):
                    dist[v] = nd; pred[v] = int(e); heapq.heappush(pq, (nd, v))
        if len(self.cache) > 200_000: self.cache.clear()
        out = (dist, pred) if with_pred else dist
        self.cache[key] = out
        return out

    def path_arcs(self, a, b, cutoff=5000.0):
        """Arc ids strictly between arc a and arc b (shortest path end(a) -> start(b))."""
        s, t = int(self.arcs[a, 1]), int(self.arcs[b, 0])
        if s == t: return []
        dist, pred = self.dijkstra(s, cutoff, True)
        if t not in dist: return None
        path = []
        while t != s:
            e = pred[t]; path.append(e); t = int(self.arcs[e, 0])
        return path[::-1]


def decode(trip, net, cand, prob, params: HMMParams):
    """Viterbi decode. cand: dict(obs, arc, u); prob: classifier score per candidate row.
    Returns predicted arc per GPS observation (n,), -1 only if the trip has no candidate at all."""
    n = len(trip.xy)
    obs, arc, u = cand["obs"], cand["arc"], cand["u"]
    first = np.r_[0, np.nonzero(np.diff(obs))[0] + 1]
    last = np.r_[first[1:], len(obs)]
    steps = [(int(obs[s]), s, e) for s, e in zip(first, last)]
    P, T = trip.xy, trip.t
    arcs_nodes = trip.arcs
    L = net.len
    pred = np.full(n, -1, np.int64)
    if not steps: return pred
    prev = None   # (t, arc ids, u, score vector, backpointer list index)
    layers = []   # per step: (t, arc_arr, back_idx_arr)
    for (t, s, e) in steps:
        a_ids, uu = arc[s:e], u[s:e]
        pr = prob[s:e]; pr = np.clip(pr, 1e-6, None); pr = pr / pr.sum()
        em = params.alpha * np.log(pr)
        if prev is None:
            layers.append((t, a_ids, np.full(len(a_ids), -1), em)); prev = (t, a_ids, uu, em); continue
        pt, pa, pu, pscore = prev
        dgps = float(np.hypot(*(P[t] - P[pt])))
        if T[t] - T[pt] > params.gap_s:               # GPS gap -> restart chain
            layers.append((t, a_ids, np.full(len(a_ids), -1), em)); prev = (t, a_ids, uu, em); continue
        cutoff = min(max(dgps * params.cutoff_mult, params.cutoff_min), params.cutoff_max)
        tr = np.full((len(pa), len(a_ids)), -params.invalid_pen)
        for i in range(len(pa)):
            ai = pa[i]; dist = None
            rem = (1 - pu[i]) * L[ai]
            for j in range(len(a_ids)):
                aj = a_ids[j]
                if aj == ai:
                    delta = uu[j] - pu[i]
                    if delta < -params.back_tol: continue          # large backward move on same arc: invalid
                    dnet = abs(delta) * L[ai]
                else:
                    if dist is None: dist = net.dijkstra(int(arcs_nodes[ai, 1]), cutoff)
                    dd = dist.get(int(arcs_nodes[aj, 0]))
                    if dd is None: continue
                    dnet = rem + dd + uu[j] * L[aj]
                    if arcs_nodes[ai, 0] == arcs_nodes[aj, 1] and arcs_nodes[ai, 1] == arcs_nodes[aj, 0]:
                        tr[i, j] = -abs(dnet - dgps) / params.beta - params.uturn_pen; continue
                tr[i, j] = -abs(dnet - dgps) / params.beta
        tot = pscore[:, None] + tr
        bp = tot.argmax(0)
        score = tot[bp, np.arange(len(a_ids))] + em
        layers.append((t, a_ids, bp, score)); prev = (t, a_ids, uu, score)
    # backtrack (restarts: back idx -1 means chain restarted -> pick argmax of that segment)
    k = int(np.argmax(layers[-1][3]))
    for li in range(len(layers) - 1, -1, -1):
        t, a_ids, bp, sc = layers[li]
        pred[t] = a_ids[k]
        if bp[k] >= 0: k = int(bp[k])
        elif li > 0: k = int(np.argmax(layers[li - 1][3]))
    # fill observations without candidates
    idx = np.nonzero(pred >= 0)[0]
    if len(idx):
        fill = np.searchsorted(idx, np.arange(n), side="right") - 1
        fill = np.clip(fill, 0, len(idx) - 1)
        miss = pred < 0
        pred[miss] = pred[idx[fill[miss]]]
        # for leading observations, searchsorted gives -1 -> clip to first decoded arc
    return pred
