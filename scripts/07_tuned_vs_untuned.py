"""Same metrics, same partitions for untuned vs Bayesian-tuned LightGBM:
CV macro-F1 / log loss / top-1 (5-fold GroupKFold, 80 train trips) and student-test top-1, macro-F1, log loss, inference time."""
import json, time, numpy as np, lightgbm as lgb
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from pathsnap.metrics import top1_pred
SEED = 42
ds = load_dataset()
best = json.load(open(os.path.join(ROOT, "results/bayes_opt.json")))["best_params"]
cfg = {"untuned": dict(n_estimators=300, learning_rate=0.1, num_leaves=31),
       "tuned": dict(**best, subsample_freq=1)}
mk = lambda c: lgb.LGBMClassifier(**c, random_state=SEED, verbose=-1, n_jobs=-1)
X, y, g, o, _ = stack(ds, "train", obs_frac=0.25, seed=SEED)
Xa, ya, _, _, _ = stack(ds, "train", obs_frac=1.0)
def top1(yv, pv, ov):
    order = np.lexsort((-pv, ov)); oo = ov[order]; f = np.r_[True, oo[1:] != oo[:-1]]
    return yv[order][f].mean()
out = {}
for name, c in cfg.items():
    mf, ll, t1 = [], [], []
    for tr, va in GroupKFold(5).split(X, y, g):
        m = mk(c).fit(X[tr], y[tr]); p = m.predict_proba(X[va])[:, 1]
        mf.append(f1_score(y[va], p >= .5, average="macro")); ll.append(log_loss(y[va], np.clip(p, 1e-7, 1 - 1e-7))); t1.append(top1(y[va], p, o[va]))
    t0 = time.time(); model = mk(c).fit(Xa, ya); fit = time.time() - t0
    P, Y, n_tot, c1, inf = [], [], 0, 0, 0.0
    for k, v in sorted((k, v) for k, v in ds.items() if k[0] == "student_test"):
        t0 = time.time(); p = model.predict_proba(v["X"])[:, 1]; inf += time.time() - t0
        pred = top1_pred(v["obs"], v["arc"], p, v["n"]); c1 += (pred == v["ref"]).sum(); n_tot += v["n"]; P.append(p); Y.append(v["y"])
    P, Y = np.concatenate(P), np.concatenate(Y)
    out[name] = dict(cv_macro_f1=float(np.mean(mf)), cv_macro_f1_std=float(np.std(mf)), cv_logloss=float(np.mean(ll)), cv_top1=float(np.mean(t1)),
                     student_top1=float(c1 / n_tot), student_macro_f1=float(f1_score(Y, P >= .5, average="macro")),
                     student_logloss=float(log_loss(Y, np.clip(P, 1e-7, 1 - 1e-7))), inference_sec=inf, fit_sec_all_train=fit, config=c)
    print(name, {k: round(v, 4) for k, v in out[name].items() if k != "config"}, flush=True)
json.dump(out, open(os.path.join(ROOT, "results/tuned_vs_untuned.json"), "w"), indent=1)
