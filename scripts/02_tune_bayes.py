"""Bayesian (TPE, Optuna) hyperparameter optimisation of LightGBM with group-aware CV (80 training trips).
Objective: minimise mean grouped-CV log loss (probability quality matters because the HMM consumes probabilities);
observation top-1 is reported alongside. Student-test is only used for the tuned-vs-untuned comparison."""
import sys, json, time, optuna, numpy as np, lightgbm as lgb
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
ds = load_dataset()
X, y, g, o, _ = stack(ds, "train", obs_frac=0.25, seed=SEED)
folds = list(GroupKFold(5).split(X, y, g))
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40

def top1(yv, pv, ov):
    order = np.lexsort((-pv, ov)); oo = ov[order]; f = np.r_[True, oo[1:] != oo[:-1]]
    return yv[order][f].mean()

def objective(trial):
    p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
             learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
             min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
             subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
             random_state=SEED, verbose=-1, n_jobs=-1)
    lls, t1s = [], []
    for tr, va in folds:
        m = lgb.LGBMClassifier(**p).fit(X[tr], y[tr]); pv = m.predict_proba(X[va])[:, 1]
        lls.append(log_loss(y[va], np.clip(pv, 1e-7, 1 - 1e-7))); t1s.append(top1(y[va], pv, o[va]))
    trial.set_user_attr("cv_top1", float(np.mean(t1s)))
    return float(np.mean(lls))

t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
study.optimize(objective, n_trials=N_TRIALS, callbacks=[lambda s, t: print(f"trial {t.number:2d} ll={t.value:.4f} top1={t.user_attrs['cv_top1']:.4f} best={s.best_value:.4f}", flush=True)])
cost = time.time() - t0
best = study.best_params
trials = [dict(number=t.number, value=t.value, cv_top1=t.user_attrs["cv_top1"], params=t.params) for t in study.trials]
json.dump(dict(best_params=best, best_cv_logloss=study.best_value, n_trials=N_TRIALS, seconds=cost, trials=trials),
          open(os.path.join(ROOT, "results/bayes_opt.json"), "w"), indent=1)
print("best", best, "cost %.0fs" % cost)
