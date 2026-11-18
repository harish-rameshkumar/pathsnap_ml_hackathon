"""Route Mismatch Fraction (RMF, Newson & Krumm 2009), as used in Fiedler et al. (arXiv:1910.05312):
RMF = (l_minus + l_plus) / l_gt, with lengths (m) of reference arcs missed (l_minus), predicted arcs not in the
reference (l_plus) and the whole reference route (l_gt). Predicted route = set of arcs predicted for the GPS fixes.
Usage: python scripts/14_rmf.py   (student-test; set EVAL_DIR to add the evaluator trips)"""
import os, numpy as np, pandas as pd
from common import *
from pathsnap.geo import list_trips, load_trip
from pathsnap.pipeline import Predictor
from pathsnap.hmm import Net
pred = Predictor(os.path.join(ROOT, "artifacts")); rows = []
parts = [("student_test", os.path.join(DATA_ROOT, "student_test"))]
if os.environ.get("EVAL_DIR"): parts.append(("evaluator_test", os.environ["EVAL_DIR"]))
for part, d in parts:
    for tid in list_trips(d):
        res = pred.predict(os.path.join(d, tid)); trip = res["trip"]
        ref = set(load_trip(os.path.join(d, tid)).route.tolist()); L = res["net"].len
        gt = L[list(ref)].sum(); r = dict(partition=part, trip=tid, gt_len_km=gt / 1000)
        for tag in ("clf", "hmm"):
            p = set(res[tag][res[tag] >= 0].tolist())
            lm, lp = L[list(ref - p)].sum(), (L[list(p - ref)].sum() if p - ref else 0.0)
            r[f"{tag}_rmf"] = (lm + lp) / gt
        rows.append(r); print(part, tid, round(r["clf_rmf"], 3), round(r["hmm_rmf"], 3), flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "results/rmf.csv"), index=False)
print(df.groupby("partition")[["clf_rmf", "hmm_rmf"]].mean())
