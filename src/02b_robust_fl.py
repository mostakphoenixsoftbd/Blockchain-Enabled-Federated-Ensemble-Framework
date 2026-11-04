"""Byzantine-robust FL comparator (coordinate-wise median FedAvg), clean and under A1.

usage: python 02b_robust_fl.py --seed 0
Writes results/runs/robust_p0.85_s{seed}.json and _preds.npz (same pooled-test order as 02_run.py).
"""
import argparse
import json
import time
import warnings

import numpy as np

from common import RESULTS, SITES, load_cohort, build_features, metrics, prepare
from fedlib import Preproc, fed_threshold, mlp_predict, train_fed_mlp

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--purity", type=float, default=0.85)
args = ap.parse_args()
P, SEED = args.purity, args.seed
OUT = RESULTS / "runs"
tag = f"robust_p{P:.2f}_s{SEED}"
t_all = time.perf_counter()

d = load_cohort()
X, y, site, split = prepare(P, SEED, d, build_features(d))
del d
cols, Xv = X.columns, X.values
I = {(s, sp): np.where((site == s) & (split == sp))[0] for s in SITES for sp in ("train", "val", "test")}
Xs = {k: Xv[i] for k, i in I.items()}
ys = {k: y[i] for k, i in I.items()}
test_idx = np.concatenate([I[s, "test"] for s in SITES])
test_site, yt = site[test_idx], y[test_idx]

pre = Preproc(cols).fit_federated([Xs[s, "train"] for s in SITES])
Z = {k: pre.transform(v) for k, v in Xs.items()}
Zval = {s: Z[s, "val"] for s in SITES}; yval = {s: ys[s, "val"] for s in SITES}
Ztest = pre.transform(Xv[test_idx])
site_data = {s: (Z[s, "train"], ys[s, "train"], Zval[s], yval[s]) for s in SITES}

res = {"purity": P, "seed": SEED, "methods": {}, "per_site": {}, "timing": {}}
preds = {}
for name, attacker in (("FedMedian-MLP", None), ("A1-FedMedian-MLP", "D")):
    t = time.perf_counter()
    fm, hist = train_fed_mlp(site_data, SEED, attacker=attacker, agg="median")
    res["timing"][f"{name}_s"] = time.perf_counter() - t
    p = mlp_predict(fm, Ztest)
    thr = fed_threshold(lambda A: mlp_predict(fm, A), Zval, yval)
    preds[name] = p.astype(np.float32)
    res["methods"][name] = metrics(yt, p, thr)
    res["per_site"][name] = {s: metrics(yt[test_site == s], p[test_site == s], thr) for s in SITES}
    res[f"{name}_val_loss_curve"] = hist
    m = res["methods"][name]
    print(f"[{tag} {time.perf_counter() - t_all:6.0f}s] {name:18s} AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} "
          f"ECE {m['ece']:.4f}", flush=True)

(OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / f"{tag}_preds.npz", y=yt, site=test_site, **preds)
