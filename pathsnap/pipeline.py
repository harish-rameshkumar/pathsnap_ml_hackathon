"""Inference pipeline (no ground truth in here) + a separate server-side evaluator."""
from __future__ import annotations
import json, os, time
import joblib, numpy as np
from .geo import load_trip, list_trips
from .candidates import generate, features
from .hmm import Net, decode, HMMParams
from .metrics import top1_pred, continuity, route_f1, pointwise
from .labels import align_to_route


class Predictor:
    def __init__(self, artifact_dir):
        bundle = joblib.load(os.path.join(artifact_dir, "model.joblib"))
        self.model = bundle["model"]
        self.params = HMMParams(**json.load(open(os.path.join(artifact_dir, "hmm_config.json"))))

    def predict(self, trip_dir):
        """Uses ONLY .track/.nodes/.arcs (route file is never read)."""
        T = {}
        t0 = time.time(); trip = load_trip(trip_dir, with_route=False); T["load"] = time.time() - t0
        t0 = time.time(); cand = generate(trip); X = features(trip, cand); T["candidates_features"] = time.time() - t0
        t0 = time.time(); p = self.model.predict_proba(X)[:, 1]; T["classifier"] = time.time() - t0
        clf = top1_pred(cand["obs"], cand["arc"], p, len(trip.xy))
        t0 = time.time(); net = Net(trip.arcs, trip.nodes_xy)
        hmm = decode(trip, net, cand, p, self.params); T["hmm"] = time.time() - t0
        return dict(trip=trip, cand=cand, net=net, clf=clf, hmm=hmm, timing=T)

    @staticmethod
    def route_polyline(res):
        """Connected predicted route as [[lat, lon], ...] (gaps filled with shortest paths)."""
        trip, net, pred = res["trip"], res["net"], res["hmm"]
        seq = [int(a) for i, a in enumerate(pred) if a >= 0 and (i == 0 or a != pred[i - 1])]
        full = []
        for i, a in enumerate(seq):
            if i:
                fill = net.path_arcs(seq[i - 1], a)
                if fill: full.extend(fill)
            full.append(a)
        pts = []
        for a in full:
            f, t = trip.arcs[a]
            for nid in (f, t):
                ll = trip.nodes_ll[nid]
                q = [float(ll[1]), float(ll[0])]
                if not pts or pts[-1] != q: pts.append(q)
        return pts, full


def evaluate(trip_dir, res):
    """Server-side only: compares predictions to the reference route."""
    trip = load_trip(trip_dir, with_route=True)
    ref = align_to_route(trip)
    out = {}
    for tag in ("clf", "hmm"):
        pr = res[tag]
        p, r, f1 = route_f1(pr, np.unique(ref))
        out[tag] = dict(pointwise_accuracy=pointwise(pr, ref), route_continuity=continuity(pr, trip.arcs),
                        route_precision=p, route_recall=r, route_f1=f1)
    return out
