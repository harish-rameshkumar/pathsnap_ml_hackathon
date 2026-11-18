"""Graphs describing the candidate-generation stage (all data from artifacts/dataset.pkl and the raw trips)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
from pathsnap.candidates import FEATURES, generate, RADIUS, K_MAX
from pathsnap.geo import load_trip
ds = load_dataset(); ix = {f: i for i, f in enumerate(FEATURES)}
F = os.path.join(ROOT, "results/figures")
train = [v for k, v in ds.items() if k[0] == "train" and v["miss"] <= MISS_EXCLUDE]
n_cand = np.concatenate([np.bincount(v["obs"], minlength=v["n"]) for v in train])
X = np.vstack([v["X"] for v in train]); y = np.concatenate([v["y"] for v in train])
d, rank = X[:, ix["d"]], X[:, ix["rank"]]
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0, 0].hist(n_cand, bins=np.arange(0, K_MAX + 2) - .5); ax[0, 0].set_title(f"Candidates per GPS fix (cap {K_MAX}, radius {RADIUS:.0f} m)"); ax[0, 0].set_xlabel("# candidates"); ax[0, 0].set_ylabel("fixes")
ax[0, 1].hist(d[y == 1], 60, alpha=.7, density=True, label="reference arc"); ax[0, 1].hist(d[y == 0], 60, alpha=.5, density=True, label="other candidates")
ax[0, 1].set_title("Distance fix -> candidate arc (m)"); ax[0, 1].legend(); ax[0, 1].set_xlabel("metres")
rr = np.bincount(rank[y == 1].astype(int), minlength=K_MAX)[:K_MAX]
ax[0, 2].bar(range(K_MAX), rr / rr.sum()); ax[0, 2].set_title("Distance-rank of the reference arc (0 = nearest)\nonly ~57% nearest; rank 1 is often the opposite-direction arc"); ax[0, 2].set_xlabel("rank by distance"); ax[0, 2].set_ylabel("share")
names = ["cos_head", "d_smooth_rel", "supp_prev"]
ax[1, 0].hist(X[y == 1, ix["cos_head"]], 50, alpha=.7, density=True, label="reference"); ax[1, 0].hist(X[y == 0, ix["cos_head"]], 50, alpha=.5, density=True, label="other"); ax[1, 0].set_title("Heading agreement (cos) of candidate"); ax[1, 0].legend()
ax[1, 1].hist(np.clip(X[y == 1, ix["d_smooth_rel"]], 0, 40), 50, alpha=.7, density=True, label="reference"); ax[1, 1].hist(np.clip(X[y == 0, ix["d_smooth_rel"]], 0, 40), 50, alpha=.5, density=True, label="other"); ax[1, 1].set_title("Smoothed distance minus best candidate's (m, clipped at 40)"); ax[1, 1].legend()
keys = sorted(k for k in ds if k[0] in ("train", "student_test")); miss = [ds[k]["miss"] * 100 for k in keys]
cols = ["tab:red" if m / 100 > MISS_EXCLUDE else ("tab:orange" if k[0] == "student_test" else "tab:blue") for k, m in zip(keys, miss)]
ax[1, 2].bar(range(len(keys)), miss, color=cols); ax[1, 2].set_yscale("symlog", linthresh=1); ax[1, 2].set_title("Candidate miss rate per trip (%)\nblue=train, orange=student-test, red=excluded (>30%)"); ax[1, 2].set_xlabel("trip index")
plt.tight_layout(); plt.savefig(F + "/candidates_overview.png", dpi=110); plt.close()
# candidate geometry example: 4 fixes with all their candidate arcs, reference arc highlighted
trip = load_trip(os.path.join(DATA_ROOT, "student_test", "00000056")); cand = generate(trip); v = ds[("student_test", "00000056")]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.8)); N = trip.nodes_xy; picks = [100, 800, 2000, 3200]
for ax_, t in zip(axes, picks):
    m = cand["obs"] == t; p = trip.xy[t]
    for a in cand["arc"][m]:
        s, e = N[trip.arcs[a, 0]], N[trip.arcs[a, 1]]; ax_.plot([s[0], e[0]], [s[1], e[1]], color="#e11d48" if a == v["ref"][t] else "#888", lw=3 if a == v["ref"][t] else 1.5)
    ax_.plot(*p, "b*", ms=14); ax_.add_patch(plt.Circle(p, RADIUS, fill=False, ls="--", color="b", lw=.8)); ax_.set_aspect("equal")
    ax_.set_title(f"fix {t}: {m.sum()} candidates (red = reference)", fontsize=9); ax_.set_xlim(p[0] - 120, p[0] + 120); ax_.set_ylim(p[1] - 120, p[1] + 120)
plt.tight_layout(); plt.savefig(F + "/candidates_examples.png", dpi=110)
print("share ref arc is nearest:", float(rr[0] / rr.sum()), "| mean cands/fix:", float(n_cand.mean()), "| fixes with 0 cands:", float((n_cand == 0).mean()))
