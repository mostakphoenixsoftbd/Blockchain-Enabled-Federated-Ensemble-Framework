"""Sensitivity analysis: richer first-24h features (ICU module, prescriptions, microbiology orders).

usage: python 04c_ext_features.py [ext|ext2]

Pairs every {fs}_p0.85_s{seed} run with the primary p0.85_s{seed} run (same patients, partitions
and splits) and writes, to results/tables/:
  {fs}_feature_prevalence.csv   share of admissions with each extended feature, by site
  {fs}_vs_base.csv              per-method AUROC/AUPRC/ECE, base vs extended, paired differences
  {fs}_delong_holm.csv          DeLong tests of BFE vs comparators on the extended set (Holm over all)
  {fs}_site_calibration.csv     per-site O/E and sensitivity, base vs extended
  {fs}_icu_subgroup.csv         AUROC in admissions with / without ICU care in the first window
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import RESULTS, SITES, delong_test, load_cohort, partition, split_subjects

FS = sys.argv[1] if len(sys.argv) > 1 else "ext"
RUNS, TAB = RESULTS / "runs", RESULTS / "tables"
P = 0.85
METHODS = ["Central-XGB", "Local-XGB", "FedAvg-MLP", "FedProx-MLP", "Ens-Uniform", "BFE-PW", "BFE-Stack"]
PAIRS = [("BFE-PW", "FedAvg-MLP"), ("BFE-PW", "FedProx-MLP"), ("BFE-PW", "Local-XGB"), ("BFE-PW", "Central-XGB"),
         ("BFE-Stack", "FedAvg-MLP"), ("BFE-Stack", "FedProx-MLP"), ("BFE-Stack", "Local-XGB"),
         ("BFE-Stack", "Central-XGB")]
SITE_METHODS = ["BFE-PW", "BFE-Stack", "FedAvg-MLP", "FedProx-MLP", "Central-XGB"]

seeds = sorted(int(f.stem.rsplit("_s", 1)[1]) for f in RUNS.glob(f"{FS}_p{P:.2f}_s*.json"))
assert seeds, f"no {FS} runs found"
d = load_cohort(FS)
ext_cols = [c for c in d.columns if c.startswith(("icu_", "rx_", "micro_", "vs_"))]

rows, dl, sc, icu = [], [], [], []
for seed in seeds:
    site = partition(d, P, seed).values
    split = split_subjects(d, pd.Series(site), seed).values
    tidx = np.concatenate([np.where((site == s) & (split == "test"))[0] for s in SITES])
    icu_flag = d["icu_in_window"].values[tidx] == 1
    if seed == seeds[0]:
        prev = d[ext_cols].gt(0).groupby(site).mean().T
        prev["All"] = d[ext_cols].gt(0).mean()
        prev.loc["in-hospital mortality"] = list(d.label.groupby(site).mean()) + [d.label.mean()]
        prev.to_csv(TAB / f"{FS}_feature_prevalence.csv")
    run = {}
    for fs, stem in (("base", f"p{P:.2f}_s{seed}"), (FS, f"{FS}_p{P:.2f}_s{seed}")):
        run[fs] = (json.loads((RUNS / f"{stem}.json").read_text()), np.load(RUNS / f"{stem}_preds.npz", allow_pickle=True))
    yb, ye = run["base"][1]["y"], run[FS][1]["y"]
    assert np.array_equal(yb, ye) and np.array_equal(run["base"][1]["site"], run[FS][1]["site"]), "runs not paired"
    assert np.array_equal(yb, d.label.values[tidx]), "test order differs from cohort"
    y, tsite = ye, run[FS][1]["site"]
    for m in METHODS:
        r = {"seed": seed, "method": m}
        for fs in ("base", FS):
            mt = run[fs][0]["methods"][m]
            for k in ("auroc", "auprc", "ece", "brier"):
                r[f"{k}_{fs}"] = mt[k]
        rows.append(r)
    Pe = run[FS][1]
    for a, b in PAIRS:
        a1, a2, z, pv = delong_test(y, Pe[a], Pe[b])
        dl.append({"seed": seed, "comparison": f"{a} − {b}", "diff": a1 - a2, "p": pv})
    for m in SITE_METHODS:
        for fs in ("base", FS):
            js, pr = run[fs]
            for s in SITES:
                mk = tsite == s
                sc.append({"seed": seed, "method": m, "features": fs, "site": s,
                           "o_e": y[mk].mean() / pr[m][mk].mean(),
                           "sensitivity": js["per_site"][m][s]["sensitivity"],
                           "auroc": js["per_site"][m][s]["auroc"]})
        for fs in ("base", FS):
            pr = run[fs][1][m]
            for g, mk in (("ICU in window", icu_flag), ("No ICU in window", ~icu_flag)):
                icu.append({"seed": seed, "method": m, "features": fs, "group": g, "n": int(mk.sum()),
                            "events": int(y[mk].sum()), "auroc": roc_auc_score(y[mk], pr[mk])})
    print("seed", seed, "done", flush=True)

df = pd.DataFrame(rows)
for k in ("auroc", "auprc", "ece"):
    df[f"d_{k}"] = df[f"{k}_{FS}"] - df[f"{k}_base"]
agg = {}
for k in ("auroc", "auprc", "ece"):
    agg |= {f"{k}_base_mean": (f"{k}_base", "mean"), f"{k}_base_sd": (f"{k}_base", "std"),
            f"{k}_{FS}_mean": (f"{k}_{FS}", "mean"), f"{k}_{FS}_sd": (f"{k}_{FS}", "std"),
            f"d_{k}_mean": (f"d_{k}", "mean"), f"d_{k}_min": (f"d_{k}", "min"), f"d_{k}_max": (f"d_{k}", "max")}
summ = df.groupby("method", sort=False).agg(**agg).reset_index()
summ["n_seeds"] = len(seeds)
summ.to_csv(TAB / f"{FS}_vs_base.csv", index=False)
print(summ[["method", "auroc_base_mean", f"auroc_{FS}_mean", "d_auroc_mean", "auprc_base_mean",
            f"auprc_{FS}_mean", "d_auprc_mean", f"ece_{FS}_mean"]].round(4).to_string())

dd = pd.DataFrame(dl)
order_ = np.argsort(dd.p.values); m_ = len(dd); adj = np.empty(m_); running = 0.0
for rank, i in enumerate(order_):
    running = max(running, min(1.0, (m_ - rank) * dd.p.values[i]))
    adj[i] = running
dd["p_holm"] = adj
dd.to_csv(TAB / f"{FS}_delong_holm.csv", index=False)
print(dd.groupby("comparison", sort=False).agg(diff=("diff", "mean"), dmin=("diff", "min"), dmax=("diff", "max"),
                                               sig_holm=("p_holm", lambda v: int((v < 0.05).sum()))).round(4).to_string())

s_ = pd.DataFrame(sc).groupby(["method", "features", "site"], sort=False)[["o_e", "sensitivity", "auroc"]].mean()
s_ = s_.unstack("features")
s_.to_csv(TAB / f"{FS}_site_calibration.csv")
print(s_.round(3).to_string())

ic = pd.DataFrame(icu).groupby(["method", "group", "features"], sort=False).agg(
    n=("n", "mean"), events=("events", "mean"), auroc=("auroc", "mean"), auroc_sd=("auroc", "std")).reset_index()
ic.to_csv(TAB / f"{FS}_icu_subgroup.csv", index=False)
print(ic.round(4).to_string())
