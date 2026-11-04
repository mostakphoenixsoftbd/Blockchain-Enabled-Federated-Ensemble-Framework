"""Revision analyses requested in review.

1. Patient-clustered bootstrap of AUROC differences with a non-inferiority margin (post hoc).
2. Holm-adjusted DeLong p-values across all comparisons and seeds.
3. Per-site calibration intercept and slope (logistic recalibration of the outcome on logit risk).
4. Byzantine-robust FL comparator (coordinate-wise median FedAvg) summary, clean and under A1.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import RESULTS, SITES, load_cohort, build_features, prepare

RUNS, TAB = RESULTS / "runs", RESULTS / "tables"
PRIMARY, MARGIN, NBOOT = 0.85, 0.01, 500
PAIRS = [("BFE-PW", "FedAvg-MLP"), ("BFE-Stack", "FedAvg-MLP"), ("BFE-Stack", "FedProx-MLP"),
         ("BFE-PW", "FedProx-MLP"), ("BFE-Stack", "Local-XGB"), ("BFE-Stack", "Central-XGB")]
CAL = ["BFE-PW", "BFE-Stack", "FedAvg-MLP", "FedProx-MLP", "Local-XGB", "Central-XGB"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cal_int_slope(y, p):
    """Calibration slope: coefficient of logit(p) in a logistic model of y; calibration-in-the-large:
    intercept with logit(p) as offset (one Newton fit, unpenalised)."""
    z = logit(p)
    slope = LogisticRegression(C=1e6, max_iter=200).fit(z[:, None], y).coef_[0, 0]
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(z + a)))
        step = (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    return a, slope


d = load_cohort()
F = build_features(d)
boot, cal = [], []
for seed in range(5):
    _, yy, ss, sp = prepare(PRIMARY, seed, d, F)
    tidx = np.concatenate([np.where((ss == s) & (sp == "test"))[0] for s in SITES])
    P = np.load(RUNS / f"p{PRIMARY:.2f}_s{seed}_preds.npz", allow_pickle=True)
    y, site = P["y"], P["site"]
    assert np.array_equal(y, np.asarray(yy)[tidx])
    subj = d.subject_id.values[tidx]
    # cluster bootstrap: resample patients, keep all their admissions
    uniq, inv = np.unique(subj, return_inverse=True)
    rows_of = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    diffs = {pr: [] for pr in PAIRS}
    for _ in range(NBOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        i = np.concatenate([rows_of[k] for k in pick])
        yb = y[i]
        auc = {m: roc_auc_score(yb, P[m][i]) for m in {x for pr in PAIRS for x in pr}}
        for a, b in PAIRS:
            diffs[a, b].append(auc[a] - auc[b])
    for a, b in PAIRS:
        v = np.array(diffs[a, b])
        boot.append({"seed": seed, "comparison": f"{a} − {b}",
                     "diff": roc_auc_score(y, P[a]) - roc_auc_score(y, P[b]),
                     "ci_lo": np.percentile(v, 2.5), "ci_hi": np.percentile(v, 97.5),
                     "noninferior": bool(np.percentile(v, 2.5) > -MARGIN)})
    for m in CAL:
        for s in SITES + ["Pooled"]:
            mk = np.ones(len(y), bool) if s == "Pooled" else site == s
            a_, b_ = cal_int_slope(y[mk], P[m][mk])
            cal.append({"seed": seed, "method": m, "site": s, "intercept": a_, "slope": b_,
                        "o_e": y[mk].mean() / P[m][mk].mean()})
    print("seed", seed, "done", flush=True)

bt = pd.DataFrame(boot)
bt.to_csv(TAB / "noninferiority_cluster_bootstrap.csv", index=False)
bs = bt.groupby("comparison").agg(diff=("diff", "mean"), ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean"),
                                  noninferior_seeds=("noninferior", "sum")).reset_index()
bs.to_csv(TAB / "noninferiority_summary.csv", index=False)
print(bs.round(4).to_string())

cl = pd.DataFrame(cal)
cl.to_csv(TAB / "calibration_by_site_long.csv", index=False)
cs = cl.groupby(["method", "site"]).agg(intercept=("intercept", "mean"), slope=("slope", "mean"),
                                       slope_sd=("slope", "std"), o_e=("o_e", "mean")).reset_index()
cs.to_csv(TAB / "table_s_calibration_by_site.csv", index=False)
print(cs.round(3).to_string())

# Holm adjustment of DeLong p-values (all comparisons x seeds as one family)
dl = pd.read_csv(TAB / "delong_noniid.csv")
order_ = np.argsort(dl.p.values)
m = len(dl); adj = np.empty(m); running = 0.0
for rank, i in enumerate(order_):
    running = max(running, min(1.0, (m - rank) * dl.p.values[i]))
    adj[i] = running
dl["p_holm"] = adj
dl.to_csv(TAB / "delong_noniid_holm.csv", index=False)
print(dl.groupby("comparison").agg(diff=("diff", "mean"), sig_holm=("p_holm", lambda v: int((v < 0.05).sum())),
                                   p_holm_max=("p_holm", "max")).to_string())

# robust FL comparator
rob = []
for f in sorted(RUNS.glob("robust_p0.85_s*.json")):
    r = json.loads(f.read_text())
    for name, mt in r["methods"].items():
        rob.append({"seed": r["seed"], "method": name, **{k: mt[k] for k in ("auroc", "auprc", "f1", "brier", "ece")}})
if rob:
    rb = pd.DataFrame(rob)
    rs = rb.groupby("method").agg(["mean", "std"]).drop(columns="seed")
    rs.to_csv(TAB / "robust_fl_summary.csv")
    print(rs.round(4).to_string())
