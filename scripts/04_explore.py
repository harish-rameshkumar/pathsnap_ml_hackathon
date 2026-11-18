"""Dataset exploration: inventory, data-quality table, plots, folium map, road-network connectivity."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, folium
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import *
from pathsnap.geo import load_trip, list_trips
os.makedirs(os.path.join(ROOT, "results/figures"), exist_ok=True)
rows = []
for part in ("train", "student_test"):
    for tid in list_trips(os.path.join(DATA_ROOT, part)):
        d = os.path.join(DATA_ROOT, part, tid)
        tr = load_trip(d); dt = np.diff(tr.t); step = np.hypot(*np.diff(tr.xy, axis=0).T)
        N = len(tr.nodes_xy); A = coo_matrix((np.ones(len(tr.arcs)), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(N, N))
        ncomp, lab = connected_components(A, directed=False)
        deg = np.bincount(tr.arcs.ravel(), minlength=N)
        raw = np.loadtxt(os.path.join(d, tid + ".track"))
        rows.append(dict(part=part, trip=tid, n_gps=len(tr.t), n_raw_rows=len(raw), nan_rows=int((~np.isfinite(raw)).any(1).sum()),
            out_of_order=int((np.diff(raw[:, 2]) < 0).sum()), duration_s=tr.t[-1] - tr.t[0], pct_1s_steps=float((dt == 1).mean()),
            max_gap_s=dt.max(), gaps_gt10s=int((dt > 10).sum()), path_len_km=step.sum() / 1000, median_step_m=float(np.median(step)),
            max_step_m=float(step.max()), extent_km=float(np.hypot(*np.ptp(tr.xy, axis=0)) / 1000), nodes=N, arcs=len(tr.arcs),
            route_arcs=len(tr.route), components=ncomp, isolated_nodes=int((deg == 0).sum()),
            lon0=tr.lon0, lat0=tr.lat0))
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "results/dataset_inventory.csv"), index=False)
ds = load_dataset()
df["cand_miss"] = [ds[(r.part, r.trip)]["miss"] for r in df.itertuples()]
df.to_csv(os.path.join(ROOT, "results/dataset_inventory.csv"), index=False)
print(df.groupby("part")[["n_gps", "nodes", "arcs", "route_arcs", "max_gap_s", "path_len_km"]].describe().T.to_string())
print("flagged (candidate miss > 0.3):", df[df.cand_miss > 0.3][["part", "trip", "cand_miss"]].to_dict("records"))
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0, 0].hist(df.n_gps, 30); ax[0, 0].set_title("GPS points per trip")
ax[0, 1].hist(df.pct_1s_steps, 30); ax[0, 1].set_title("fraction of exactly-1 s steps")
ax[0, 2].hist(np.log10(df.max_gap_s), 30); ax[0, 2].set_title("log10 max time gap (s)")
ax[1, 0].hist(df.path_len_km, 30); ax[1, 0].set_title("GPS path length (km)")
ax[1, 1].hist(np.log10(df.arcs), 30); ax[1, 1].set_title("log10 road arcs per trip")
ax[1, 2].scatter(df.lon0, df.lat0, s=12); ax[1, 2].set_title("trip locations (lon, lat)")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "results/figures/dataset_overview.png"), dpi=110); plt.close()
# interactive folium of one trip: raw GPS + reference route
from pathsnap.labels import align_to_route
t = load_trip(os.path.join(DATA_ROOT, "train", "00000034"))
m = folium.Map(location=[t.lat0, t.lon0], zoom_start=12)
fr, to = t.arcs[t.route, 0], t.arcs[t.route, 1]
folium.PolyLine([[t.nodes_ll[a][1], t.nodes_ll[a][0]] for a in np.r_[fr[0], to]], color="green", weight=4, tooltip="reference route").add_to(m)
for p in t.lonlat[::6]: folium.CircleMarker([p[1], p[0]], radius=2, color="blue").add_to(m)
m.save(os.path.join(ROOT, "results/figures/trip_00000034_map.html"))
