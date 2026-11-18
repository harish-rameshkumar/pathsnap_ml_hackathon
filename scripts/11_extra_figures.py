import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ROOT
import os
mc = json.load(open(os.path.join(ROOT, "results/model_comparison.json"))); names = [r["model"] for r in mc]; x = np.arange(len(mc))
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
ax[0].bar(x - .2, [r["cv_macro_f1"] for r in mc], .4, yerr=[r["cv_macro_f1_std"] for r in mc], label="CV macro-F1"); ax[0].bar(x + .2, [r["macro_f1"] for r in mc], .4, label="student-test macro-F1")
ax[0].set_xticks(x, names, rotation=20, fontsize=8); ax[0].set_ylim(.4, 1); ax[0].legend(); ax[0].set_title("Macro-F1")
ax[1].bar(x - .2, [r["cv_logloss"] for r in mc], .4, label="CV log loss"); ax[1].bar(x + .2, [r["log_loss"] for r in mc], .4, label="student-test log loss")
ax[1].set_xticks(x, names, rotation=20, fontsize=8); ax[1].set_yscale("log"); ax[1].legend(); ax[1].set_title("Log loss (lower better, log scale)")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "results/figures/model_comparison_f1.png"), dpi=110)
