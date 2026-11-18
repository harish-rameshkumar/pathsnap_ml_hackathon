import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, confusion_matrix)


def top1_pred(obs, arc, score, n):
    """Per observation arc with highest score; -1 where no candidate."""
    order = np.lexsort((-score, obs))
    o, a = obs[order], arc[order]
    first = np.r_[True, o[1:] != o[:-1]]
    pred = np.full(n, -1, np.int64); pred[o[first]] = a[first]
    return pred


def continuity(pred, arcs):
    """Fraction of arc *changes* in a predicted sequence that are physically
    connected (to_node of previous arc == from_node of next arc)."""
    p = pred[pred >= 0]
    ch = p[1:] != p[:-1]
    if ch.sum() == 0:
        return 1.0
    a, b = p[:-1][ch], p[1:][ch]
    return float((arcs[a, 1] == arcs[b, 0]).mean())


def route_f1(pred, ref):
    """Set-level precision/recall/F1 between predicted and reference arc sets."""
    P, R = set(pred[pred >= 0].tolist()), set(ref.tolist())
    inter = len(P & R)
    pr = inter / max(len(P), 1); rc = inter / max(len(R), 1)
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def pointwise(pred, ref):
    return float((pred == ref).mean())


def row_metrics(y, p):
    yh = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
                confusion=confusion_matrix(y, yh, labels=[0, 1]).tolist())
