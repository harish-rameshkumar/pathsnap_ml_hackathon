"""Run the final pipeline (same code path as the API: Predictor.predict + evaluate) on the evaluator trips.
Usage: EVAL_DIR=/path/to/evaluator_test python scripts/13_evaluator_run.py
No parameter, model or threshold is changed based on these results (final-evaluation only)."""
import json, time, os, sys, numpy as np, pandas as pd
from common import *
from pathsnap.geo import list_trips
from pathsnap.pipeline import Predictor, evaluate
d = os.environ["EVAL_DIR"]; pred = Predictor(os.path.join(ROOT, "artifacts")); rows = []
for tid in list_trips(d):
    t0 = time.time(); res = pred.predict(os.path.join(d, tid)); m = evaluate(os.path.join(d, tid), res)
    rows.append(dict(trip=tid, n_gps=len(res["trip"].t), seconds=time.time() - t0,
                     **{f"{tag}_{k}": v for tag in ("clf", "hmm") for k, v in m[tag].items()}))
    print(tid, round(rows[-1]["hmm_pointwise_accuracy"], 4), flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "results/evaluator_results.csv"), index=False)
json.dump(rows, open(os.path.join(ROOT, "results/evaluator_results.json"), "w"), indent=1)
print(df.filter(like="pointwise").mean().to_dict(), df.hmm_route_continuity.mean())
