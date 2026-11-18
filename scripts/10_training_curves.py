"""Epoch/iteration-wise training histories - only for models that produce them (MLP, LightGBM, HistGradientBoosting)."""
import json, numpy as np, matplotlib, lightgbm as lgb
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
SEED = 42
ds = load_dataset()
X, y, g, o, _ = stack(ds, "train", obs_frac=0.25, seed=SEED)
tr, va = next(GroupKFold(5).split(X, y, g))
best = json.load(open(os.path.join(ROOT, "results/bayes_opt.json")))["best_params"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
sc = StandardScaler().fit(X[tr])
mlp = MLPClassifier((64, 32), max_iter=60, early_stopping=True, random_state=SEED).fit(sc.transform(X[tr]), y[tr])
ax[0].plot(mlp.loss_curve_, label="training log loss"); ax[0].set_xlabel("epoch"); ax[0].set_title("MLP(64,32): training loss")
ax2 = ax[0].twinx(); ax2.plot(mlp.validation_scores_, color="C1", label="validation accuracy (10% internal split)"); ax2.set_ylabel("val accuracy")
ax[0].legend(loc="upper center"); ax2.legend(loc="lower center")
for a, (name, p) in zip(ax[1:], [("LightGBM untuned", dict(n_estimators=300, learning_rate=0.1, num_leaves=31)), ("LightGBM tuned (Optuna)", dict(**best, subsample_freq=1))]):
    m = lgb.LGBMClassifier(**p, random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(X[tr], y[tr], eval_set=[(X[tr], y[tr]), (X[va], y[va])], eval_names=["train", "val(held-out trips)"], eval_metric="binary_logloss")
    for k, v in m.evals_result_.items(): a.plot(v["binary_logloss"], label=k)
    a.set_title(name + ": log loss per boosting round"); a.set_xlabel("boosting round"); a.legend(); a.set_ylim(0, .25)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "results/figures/training_curves.png"), dpi=110); print("ok")
