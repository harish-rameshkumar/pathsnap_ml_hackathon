"""EDA plots: missing/invalid values, GPS trajectory gallery, road-network + track visualisation, sampling-interval analysis."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
from pathsnap.geo import load_trip
F = os.path.join(ROOT, "results/figures")
inv = pd.read_csv(os.path.join(ROOT, "results/dataset_inventory.csv"))
# 1. missing / invalid value analysis (all raw .track rows, by partition)
summ = inv.groupby("part").agg(trips=("trip", "count"), raw_rows=("n_raw_rows", "sum"), nan_rows=("nan_rows", "sum"),
        out_of_order_steps=("out_of_order", "sum"), isolated_nodes=("isolated_nodes", "sum"),
        trips_with_gap_gt10s=("gaps_gt10s", lambda s: int((s > 0).sum())), median_pct_1s=("pct_1s_steps", "median")).reset_index()
summ.to_csv(os.path.join(ROOT, "results/missing_value_summary.csv"), index=False); print(summ.to_string())
# 2. sampling interval histogram over all trips
dts = []
for part in ("train", "student_test"):
    for tid in inv[inv.part == part].trip:
        t = np.loadtxt(os.path.join(DATA_ROOT, part, f"{tid:08d}" if isinstance(tid, (int, np.integer)) else tid, f"{int(tid):08d}.track"), usecols=2)
        dts.append(np.diff(t))
dts = np.concatenate(dts)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
vals, cnt = np.unique(np.clip(dts, 0, 12), return_counts=True); ax[0].bar(vals, cnt); ax[0].set_yscale("log"); ax[0].set_title("time step between fixes (s, clipped at 12) – all trips"); 
ax[1].hist(inv.pct_1s_steps, 25); ax[1].set_title("fraction of exactly-1 s steps per trip")
plt.tight_layout(); plt.savefig(F + "/sampling_intervals.png", dpi=110); plt.close()
# 3. trajectory gallery: 6 training trips, GPS only
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, tid in zip(axes.ravel(), ["00000005", "00000012", "00000034", "00000021", "00000013", "00000004"]):
    t = load_trip(os.path.join(DATA_ROOT, "train", tid)); ax.plot(t.xy[:, 0], t.xy[:, 1], ".", ms=1.5)
    ax.set_aspect("equal"); ax.set_title(f"train trip {tid}: {len(t.t)} fixes", fontsize=9)
plt.tight_layout(); plt.savefig(F + "/gps_trajectories_train.png", dpi=100); plt.close()
# 4. road network + GPS + reference route for one training trip and one student trip
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (part, tid) in zip(axes, [("train", "00000034"), ("student_test", "00000040")]):
    t = load_trip(os.path.join(DATA_ROOT, part, tid)); pad = 300
    lo, hi = t.xy.min(0) - pad, t.xy.max(0) + pad
    A, B = t.nodes_xy[t.arcs[:, 0]], t.nodes_xy[t.arcs[:, 1]]
    m = ((np.minimum(A, B) <= hi).all(1)) & ((np.maximum(A, B) >= lo).all(1))
    for a, b in zip(A[m], B[m]): ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbb", lw=.5)
    rt = t.route
    for a, b in zip(A[rt], B[rt]): ax.plot([a[0], b[0]], [a[1], b[1]], color="green", lw=2.5, alpha=.7)
    ax.plot(t.xy[:, 0], t.xy[:, 1], ".", ms=1.5, c="blue"); ax.set_aspect("equal")
    ax.set_title(f"{part} trip {tid}: network (grey, {m.sum()} arcs in view of {len(t.arcs)}), reference route (green), GPS (blue)", fontsize=8)
plt.tight_layout(); plt.savefig(F + "/road_network_gps_route.png", dpi=100)
# 5. network size / connectivity summary
inv[["part", "trip", "nodes", "arcs", "route_arcs", "components", "isolated_nodes"]].to_csv(os.path.join(ROOT, "results/network_connectivity.csv"), index=False)
print(inv.groupby("part")[["nodes", "arcs", "components", "isolated_nodes"]].describe().T.round(1).to_string()[:1500])
