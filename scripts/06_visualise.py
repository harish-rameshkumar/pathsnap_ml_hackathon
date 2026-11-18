"""Report visualisations: predicted vs reference routes (static PNG + interactive folium) and confusion matrices."""
import json, numpy as np, matplotlib  # interactive Folium maps live in 06b_folium_maps.py
matplotlib.use("Agg"); import matplotlib.pyplot as plt, folium
from common import *
from pathsnap.geo import load_trip
from pathsnap.pipeline import Predictor
from pathsnap.labels import align_to_route
from pathsnap.metrics import pointwise
F = os.path.join(ROOT, "results/figures"); os.makedirs(F, exist_ok=True)
pred = Predictor(os.path.join(ROOT, "artifacts"))
TRIPS = ["00000007", "00000056", "00000077"]   # best-ish, mid, worst student trip
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, tid in zip(axes, TRIPS):
    d = os.path.join(DATA_ROOT, "student_test", tid)
    res = pred.predict(d); trip = res["trip"]
    ref_trip = load_trip(d); ref = align_to_route(ref_trip)
    xy, N = trip.xy, trip.nodes_xy
    seg = lambda arcs: [(N[trip.arcs[a, 0]], N[trip.arcs[a, 1]]) for a in arcs]
    for a, b in seg(np.unique(ref_trip.route)): ax.plot([a[0], b[0]], [a[1], b[1]], color="green", lw=4, alpha=.5)
    for a, b in seg(np.unique(res["hmm"][res["hmm"] >= 0])): ax.plot([a[0], b[0]], [a[1], b[1]], color="red", lw=1.5)
    wrong = res["hmm"] != ref
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c="blue", alpha=.5); ax.scatter(xy[wrong, 0], xy[wrong, 1], s=8, c="orange")
    ax.set_aspect("equal"); ax.set_title(f"trip {tid}: HMM acc {pointwise(res['hmm'], ref):.3f}\ngreen=reference, red=predicted, blue=GPS, orange=mismatched fixes", fontsize=9)
plt.tight_layout(); plt.savefig(F + "/routes_pred_vs_ref.png", dpi=110); plt.close()
# confusion matrices (student-test, row level, threshold 0.5)
mc = json.load(open(os.path.join(ROOT, "results/model_comparison.json")))
fig, axes = plt.subplots(3, 3, figsize=(11, 10))
for ax, r in zip(axes.ravel(), mc):
    cm = np.array(r["confusion"]); ax.imshow(np.log10(cm + 1), cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"]); ax.set_title(r["model"], fontsize=9)
plt.tight_layout(); plt.savefig(F + "/confusion_matrices.png", dpi=110)
print("ok")
