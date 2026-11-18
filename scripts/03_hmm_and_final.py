"""1) OOF classifier probabilities on train trips (group CV) with tuned LightGBM
   2) tune HMM params (Optuna) on a subset of TRAIN trips (OOF probs)
   3) fit final model on all train trips, export artifacts
   4) report classifier-only vs classifier+HMM on the 10 student-test trips"""
import json, time, joblib, optuna, numpy as np, lightgbm as lgb
from common import *
from sklearn.model_selection import GroupKFold
from pathsnap.geo import load_trip
from pathsnap.candidates import generate, FEATURES
from pathsnap.hmm import Net, decode, HMMParams
from pathsnap.metrics import *
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
ds = load_dataset()
best = json.load(open(os.path.join(ROOT, "results/bayes_opt.json")))["best_params"]
mk = lambda: lgb.LGBMClassifier(**best, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=-1)

# --- untuned baseline for the tuned-vs-untuned comparison
Xa, ya, ga, oa, keys = stack(ds, "train", obs_frac=1.0)
Xs, ys, _, _, _ = stack(ds, "student_test", exclude_bad=False)
base = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=SEED, verbose=-1, n_jobs=-1)
t0 = time.time(); base.fit(Xa, ya); t_base = time.time() - t0
t0 = time.time(); final = mk().fit(Xa, ya); t_final = time.time() - t0
joblib.dump(dict(model=final, features=FEATURES), os.path.join(ROOT, "artifacts/model.joblib"))

# --- OOF probs for train trips
gmap = dict(zip(np.unique(ga), keys))   # group ids skip excluded trips
oof = {}
for tr, va in GroupKFold(5).split(Xa, ya, ga):
    m = mk().fit(Xa[tr], ya[tr])
    for gi in np.unique(ga[va]):
        rows = va[ga[va] == gi]
        oof[gmap[gi]] = m.predict_proba(Xa[rows])[:, 1]
print("oof done", flush=True)

def prep(k):
    trip = load_trip(os.path.join(DATA_ROOT, *k)); cand = generate(trip)
    assert np.array_equal(cand["arc"], ds[k]["arc"])
    return trip, cand, Net(trip.arcs, trip.nodes_xy)

# tuning subset: 24 train trips with n<=4500 and reliable labels
rng = np.random.default_rng(SEED)
pool = [k for k in keys if ds[k]["n"] <= 4500]
tune_keys = [pool[i] for i in rng.choice(len(pool), 24, replace=False)]
cache = {k: prep(k) for k in tune_keys}

def run(k, prob, params, cache_):
    trip, cand, net = cache_[k]
    return decode(trip, net, cand, prob, params)

def score(params, ks, probs, cache_):
    acc = []
    for k in ks:
        pred = run(k, probs[k], params, cache_); acc.append(pointwise(pred, ds[k]["ref"]))
    return float(np.mean(acc))

def obj(t):
    p = HMMParams(alpha=t.suggest_float("alpha", 0.2, 3.0), beta=t.suggest_float("beta", 5, 150, log=True),
                  invalid_pen=t.suggest_float("invalid_pen", 2, 30), uturn_pen=t.suggest_float("uturn_pen", 0, 15),
                  gap_s=t.suggest_categorical("gap_s", [10, 30, 60, 120, 300]))
    return score(p, tune_keys, oof, cache)
t0 = time.time()
st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
st.optimize(obj, n_trials=30, callbacks=[lambda s, t: print(f"hmm trial {t.number} acc={t.value:.4f} best={s.best_value:.4f}", flush=True)])
hp = HMMParams(**st.best_params)
hmm_cost = time.time() - t0
print("best HMM", hp, flush=True)

# --- report on student test
rep = {}
for name, model in [("untuned_lgbm", base), ("tuned_lgbm", final)]:
    rows, hrows = [], []
    for k in sorted(k for k in ds if k[0] == "student_test"):
        v = ds[k]; trip, cand, net = prep(k)
        p = model.predict_proba(v["X"])[:, 1]
        pc = top1_pred(v["obs"], v["arc"], p, v["n"])
        ph = decode(trip, net, cand, p, hp)
        ref = v["ref"]
        r = dict(trip=k[1], n=v["n"])
        for tag, pr in (("clf", pc), ("hmm", ph)):
            r[tag + "_acc"] = pointwise(pr, ref); r[tag + "_cont"] = continuity(pr, trip.arcs)
            pp, rc, f1 = route_f1(pr, np.unique(ref)); r[tag + "_route_f1"] = f1; r[tag+"_route_p"]=pp; r[tag+"_route_r"]=rc
        rows.append(r); print(name, r, flush=True)
    rep[name] = rows
json.dump(dict(hmm_params=hp.to_dict(), hmm_tune_seconds=hmm_cost, hmm_tune_trips=[k[1] for k in tune_keys],
               train_sec=dict(untuned=t_base, tuned=t_final), report=rep),
          open(os.path.join(ROOT, "results/hmm_and_final.json"), "w"), indent=1)
json.dump(hp.to_dict(), open(os.path.join(ROOT, "artifacts/hmm_config.json"), "w"), indent=1)
