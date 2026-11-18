import os, sys, pickle, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA_ROOT = os.environ.get("PATHSNAP_DATA", "/Users/harish/Desktop/untitled folder/Student folder")
MISS_EXCLUDE = 0.30   # trips whose reference route doesn't match the GPS are excluded from fitting


def load_dataset():
    return pickle.load(open(os.path.join(ROOT, "artifacts/dataset.pkl"), "rb"))


def stack(ds, part, exclude_bad=True, obs_frac=1.0, seed=0):
    """Concatenate trips -> X, y, groups(trip index), plus (part,tid) list.
    obs_frac subsamples whole observations (keeps all candidates of an obs)."""
    rng = np.random.default_rng(seed)
    Xs, ys, gs, os_, keys = [], [], [], [], []
    for gi, (k, v) in enumerate(sorted((k, v) for k, v in ds.items() if k[0] == part)):
        if exclude_bad and v["miss"] > MISS_EXCLUDE:
            continue
        m = np.ones(len(v["y"]), bool)
        if obs_frac < 1:
            keep_obs = rng.random(v["n"]) < obs_frac
            m = keep_obs[v["obs"]]
        Xs.append(v["X"][m]); ys.append(v["y"][m]); gs.append(np.full(m.sum(), gi))
        os_.append(v["obs"][m] + gi * 10_000_000); keys.append(k)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(os_), keys
