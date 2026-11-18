"""Build labelled candidate tables for every trip in train/ and student_test/.
Usage: python scripts/00_build_dataset.py <data_root>"""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pathsnap.geo import load_trip, list_trips
from pathsnap.labels import align_to_route
from pathsnap.candidates import generate, features

root = sys.argv[1]
out = {}
for part in ("train", "student_test"):
    for tid in list_trips(os.path.join(root, part)):
        t0 = time.time()
        trip = load_trip(os.path.join(root, part, tid))
        ref = align_to_route(trip)
        cand = generate(trip)
        X = features(trip, cand)
        y = (cand["arc"] == ref[cand["obs"]]).astype(np.int8)
        covered = np.zeros(len(ref), bool); covered[cand["obs"][y == 1]] = True
        out[(part, tid)] = dict(X=X, y=y, obs=cand["obs"], arc=cand["arc"], ref=ref,
                                n=len(ref), miss=float(1 - covered.mean()))
        print(part, tid, "n=%d cand=%d miss=%.4f  %.1fs" % (len(ref), len(y), 1 - covered.mean(), time.time() - t0), flush=True)
os.makedirs("artifacts", exist_ok=True)
pickle.dump(out, open("artifacts/dataset.pkl", "wb"))
