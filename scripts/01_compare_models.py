"""Train + compare >=5 models from different families (group-aware CV on the 80 training trips,
development comparison on the 10 student-test trips)."""
import time, json, warnings, numpy as np
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score
from pathsnap.metrics import *
from pathsnap.geo import load_trip
warnings.filterwarnings("ignore")
SEED = 42
ds = load_dataset()
Xtr, ytr, gtr, otr, _ = stack(ds, "train", obs_frac=0.25, seed=SEED)     # experiment subsample
print("experiment set", Xtr.shape, ytr.mean())

MODELS = {
 "LogisticRegression": ("Linear/probabilistic", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, class_weight=None))),
 "GaussianNB": ("Linear/probabilistic", lambda: GaussianNB()),
 "KNN(k=25,distance)": ("Instance-based", lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(25, weights="distance", n_jobs=-1))),
 "DecisionTree(d=10)": ("Tree-based", lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=SEED)),
 "RandomForest(200)": ("Ensemble/bagging", lambda: RandomForestClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
 "ExtraTrees(200)": ("Ensemble/bagging", lambda: ExtraTreesClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
 "HistGradientBoosting": ("Boosting", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)),
 "LightGBM(default-ish)": ("Boosting", lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=SEED, verbose=-1, n_jobs=-1)),
 "MLP(64,32)": ("Neural network", lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=60, early_stopping=True, random_state=SEED))),
}

def eval_on(model, parts_key="student_test"):
    """Row metrics + observation-level metrics on a held-out partition."""
    allp, ally, miss_n, n_tot, top1_c, cont, f1s, pw = [], [], 0, 0, 0, [], [], []
    t_inf = 0.0
    for k, v in sorted((k, v) for k, v in ds.items() if k[0] == parts_key):
        t0 = time.time(); p = model.predict_proba(v["X"])[:, 1]; t_inf += time.time() - t0
        pred = top1_pred(v["obs"], v["arc"], p, v["n"])
        arcs = load_trip(os.path.join(DATA_ROOT, *k), with_route=False).arcs
        allp.append(p); ally.append(v["y"])
        pw.append(pointwise(pred, v["ref"])); cont.append(continuity(pred, arcs))
        f1s.append(route_f1(pred, np.unique(v["ref"]))[2])
        n_tot += v["n"]; top1_c += (pred == v["ref"]).sum(); miss_n += v["miss"] * v["n"]
    r = row_metrics(np.concatenate(ally), np.concatenate(allp))
    r['macro_f1'] = float(f1_score(np.concatenate(ally), (np.concatenate(allp) >= 0.5).astype(int), average='macro'))
    r.update(top1_obs=top1_c / n_tot, miss_rate=miss_n / n_tot, continuity=float(np.mean(cont)),
             route_f1=float(np.mean(f1s)), infer_sec_total=t_inf)
    return r

res = []
gkf = GroupKFold(5)
for name, (fam, mk) in MODELS.items():
    t0 = time.time()
    cv_ll, cv_top1, cv_mf1, tr_ll = [], [], [], []
    for tr, va in gkf.split(Xtr, ytr, gtr):
        m = mk().fit(Xtr[tr], ytr[tr]); p = m.predict_proba(Xtr[va])[:, 1]
        cv_ll.append(log_loss(ytr[va], np.clip(p, 1e-7, 1 - 1e-7)))
        cv_mf1.append(f1_score(ytr[va], (p >= 0.5).astype(int), average='macro'))
        tr_ll.append(log_loss(ytr[tr], np.clip(m.predict_proba(Xtr[tr])[:, 1], 1e-7, 1 - 1e-7)))
        # observation-level top1 among covered obs
        pred = top1_pred(otr[va], np.arange(len(va)), p, 0) if False else None
        o = otr[va]; order = np.lexsort((-p, o)); oo = o[order]; first = np.r_[True, oo[1:] != oo[:-1]]
        cv_top1.append(ytr[va][order][first].mean())
    t_cv = time.time() - t0
    t0 = time.time(); model = mk().fit(Xtr, ytr); t_fit = time.time() - t0
    r = eval_on(model)
    r.update(model=name, family=fam, cv_logloss=float(np.mean(cv_ll)), cv_logloss_std=float(np.std(cv_ll)),
             cv_macro_f1=float(np.mean(cv_mf1)), cv_macro_f1_std=float(np.std(cv_mf1)), fold_train_logloss=float(np.mean(tr_ll)),
             train_macro_f1=float(f1_score(ytr, (model.predict_proba(Xtr)[:, 1] >= 0.5).astype(int), average='macro')),
             cv_top1=float(np.mean(cv_top1)), cv_top1_std=float(np.std(cv_top1)), train_sec=t_fit, cv_sec=t_cv)
    res.append(r)
    print(f"{name:24s} cvLL={r['cv_logloss']:.4f}±{r['cv_logloss_std']:.4f} cvTop1={r['cv_top1']:.4f} | stud: top1={r['top1_obs']:.4f} "
          f"F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} LL={r['log_loss']:.4f} cont={r['continuity']:.3f} fit={t_fit:.1f}s inf={r['infer_sec_total']:.2f}s", flush=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "results/model_comparison.json"), "w"), indent=1)
