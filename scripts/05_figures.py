import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ROOT
import os
F = os.path.join(ROOT, "results/figures")
mc = pd.DataFrame(json.load(open(os.path.join(ROOT, "results/model_comparison.json"))))
mc[["model","family","cv_logloss","cv_logloss_std","cv_top1","top1_obs","precision","recall","f1","log_loss","continuity","train_sec","infer_sec_total"]].round(4).to_csv(os.path.join(ROOT,"results/model_comparison.csv"),index=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].barh(mc.model, mc.top1_obs); ax[0].set_xlim(.84, .9); ax[0].set_title("Student-test observation top-1")
ax[1].barh(mc.model, mc.cv_logloss, xerr=mc.cv_logloss_std); ax[1].set_xlim(0, .2); ax[1].set_title("Grouped-CV log loss (lower better)")
plt.tight_layout(); plt.savefig(F + "/model_comparison.png", dpi=110); plt.close()
b = json.load(open(os.path.join(ROOT, "results/bayes_opt.json")))
v = [t["value"] for t in b["trials"]]
plt.figure(figsize=(6, 4)); plt.plot(v, "o-", ms=3); plt.plot(np.minimum.accumulate(v)); plt.ylim(.058, .1)
plt.xlabel("trial"); plt.ylabel("CV log loss"); plt.title("Optuna TPE trials (LightGBM)"); plt.savefig(F + "/bayes_opt.png", dpi=110); plt.close()
h = json.load(open(os.path.join(ROOT, "results/hmm_and_final.json")))
r = pd.DataFrame(h["report"]["tuned_lgbm"])
x = np.arange(len(r)); plt.figure(figsize=(9, 4)); plt.bar(x - .2, r.clf_acc, .4, label="classifier only"); plt.bar(x + .2, r.hmm_acc, .4, label="classifier + HMM")
plt.xticks(x, r.trip, rotation=45); plt.ylim(.7, 1); plt.legend(); plt.title("Pointwise accuracy, student-test trips"); plt.tight_layout(); plt.savefig(F + "/clf_vs_hmm.png", dpi=110)
for name, rows in h["report"].items():
    d = pd.DataFrame(rows); print(name, d[[c for c in d.columns if c != "trip"]].mean().round(4).to_dict())
