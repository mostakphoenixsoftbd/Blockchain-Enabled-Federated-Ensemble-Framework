"""Revision statistics (2026-11).

1. Primary inference: joint patient-clustered bootstrap over the five seeds. In each of NBOOT
   replicates, patients are resampled (with all their admissions) independently in each seed's
   test set, the difference between two methods is computed per seed, and the five-seed mean is
   kept. The 95% CI is the percentile interval of that mean. Reported for AUROC and AUPRC.
2. Calibration: intercept (calibration-in-the-large), slope and integrated calibration index
   (ICI; Austin & Steyerberg 2019, lowess-smoothed observed risk), pooled and per site.
3. Seed-level summaries (mean, SD) for the revision runs: data-scarce sites (small*),
   care-era partition (year_), stealthier attacks / calibration fixes / FedLR (ext3_).

Run after all runs exist, on its own (it loads the cohort): python 04d_revision_stats.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess

from common import RESULTS, SITES, build_features, load_cohort, prepare

RUNS, TAB = RESULTS / "runs", RESULTS / "tables"
NBOOT, SEEDS = 1000, range(5)
REF = ["BFE-PW", "BFE-Stack"]
COMP = ["FedAvg-MLP", "FedProx-MLP", "Local-XGB", "Central-XGB"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib(y, p):
    z = logit(p)
    slope = LogisticRegression(C=1e6, max_iter=300).fit(z[:, None], y).coef_[0, 0]
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(z + a)))
        step = (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    sm = lowess(y, p, frac=0.3, it=0, delta=0.002 * (p.max() - p.min()), return_sorted=False)
    ici = float(np.mean(np.abs(sm - p)))
    return a, slope, ici, y.mean() / p.mean()


def test_subjects(d, F, seed, **kw):
    _, yy, ss, sp = prepare(0.85, seed, d, F, **kw)
    tidx = np.concatenate([np.where((ss == s) & (sp == "test"))[0] for s in SITES])
    return d.subject_id.values[tidx], np.asarray(yy)[tidx]


def joint_bootstrap(sets, pairs, label):
    """sets: list of (y, subj, preds-dict) for each seed."""
    rngs = [np.random.default_rng(100 + k) for k in range(len(sets))]
    clusters = []
    for y, subj, _ in sets:
        uniq, inv = np.unique(subj, return_inverse=True)
        clusters.append(np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1]))
    point = {pr: {"auroc": [], "auprc": []} for pr in pairs}
    for y, _, P in sets:
        for a, b in pairs:
            point[a, b]["auroc"].append(roc_auc_score(y, P[a]) - roc_auc_score(y, P[b]))
            point[a, b]["auprc"].append(average_precision_score(y, P[a]) - average_precision_score(y, P[b]))
    reps = {pr: {"auroc": np.zeros(NBOOT), "auprc": np.zeros(NBOOT)} for pr in pairs}
    methods = sorted({m for pr in pairs for m in pr})
    # exact weighted metrics: a cluster bootstrap replicate equals the original rows with integer
    # multiplicities, so each score is sorted once and AUROC/AP are computed from weighted counts
    prep = []
    for k, (y, subj, P) in enumerate(sets):
        uniq, inv = np.unique(subj, return_inverse=True)
        pm = {}
        for m in methods:
            o = np.argsort(-P[m], kind="stable"); ps = P[m][o]
            starts = np.r_[0, np.nonzero(np.diff(ps))[0] + 1]
            pm[m] = (o, starts)
        prep.append((np.asarray(y, float), inv, len(uniq), pm))
    for r in range(NBOOT):
        acc = {pr: {"auroc": 0.0, "auprc": 0.0} for pr in pairs}
        for k, (y, inv, nc, pm) in enumerate(prep):
            cw = np.bincount(rngs[k].integers(0, nc, nc), minlength=nc).astype(float)
            w = cw[inv]
            au, ap = {}, {}
            for m in methods:
                o, st = pm[m]
                wo, yo = w[o], y[o]
                Pg = np.add.reduceat(wo * yo, st); Ng = np.add.reduceat(wo * (1 - yo), st)
                Pt, Nt = Pg.sum(), Ng.sum()
                cP, cN = np.cumsum(Pg), np.cumsum(Ng)
                au[m] = float(np.sum(Pg * (Nt - cN) + 0.5 * Pg * Ng) / (Pt * Nt))
                with np.errstate(invalid="ignore", divide="ignore"):
                    prec = np.where(cP + cN > 0, cP / (cP + cN), 0.0)
                ap[m] = float(np.sum(Pg * prec) / Pt)
            for a, b in pairs:
                acc[a, b]["auroc"] += (au[a] - au[b]) / len(sets)
                acc[a, b]["auprc"] += (ap[a] - ap[b]) / len(sets)
        for pr in pairs:
            for mt in ("auroc", "auprc"):
                reps[pr][mt][r] = acc[pr][mt]
        if (r + 1) % 100 == 0:
            print(label, "bootstrap", r + 1, flush=True)
    out = []
    for a, b in pairs:
        for mt in ("auroc", "auprc"):
            v = reps[a, b][mt]
            out.append({"set": label, "comparison": f"{a} − {b}", "metric": mt,
                        "diff_mean": float(np.mean(point[a, b][mt])), "diff_sd_seeds": float(np.std(point[a, b][mt], ddof=1)),
                        "ci_lo": float(np.percentile(v, 2.5)), "ci_hi": float(np.percentile(v, 97.5)),
                        "seeds_positive": int(sum(x > 0 for x in point[a, b][mt]))})
    return out


def load_preds(path, names):
    P = np.load(path, allow_pickle=True)
    return P["y"], P["site"], {m: P[m] for m in names if m in P.files}


def seed_summary(prefix, label):
    rows = []
    for f in sorted(RUNS.glob(f"{prefix}p0.85_s*.json")):
        r = json.loads(f.read_text())
        for name, mt in r["methods"].items():
            rows.append({"set": label, "seed": r["seed"], "method": name,
                         **{k: mt[k] for k in ("auroc", "auprc", "f1", "sensitivity", "brier", "ece") if k in mt}})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df.groupby(["set", "method"]).agg(n_seeds=("seed", "nunique"),
                                             **{f"{k}_{s}": (k, s) for k in ("auroc", "auprc", "brier", "ece")
                                                for s in ("mean", "std")}).reset_index()


if __name__ == "__main__":
    d = load_cohort()
    F = build_features(d)
    boot, cal = [], []

    # ---- main configuration (simulated hospitals, purity 0.85)
    sets = []
    for s in SEEDS:
        subj, yy = test_subjects(d, F, s)
        y, site, P = load_preds(RUNS / f"p0.85_s{s}_preds.npz", REF + COMP)
        assert np.array_equal(y, yy)
        rb = np.load(RUNS / f"robust_p0.85_s{s}_preds.npz", allow_pickle=True)
        P["FedMedian-MLP"] = rb["FedMedian-MLP"]
        sets.append((y, subj, P))
        for m in REF + COMP + ["FedMedian-MLP"]:
            for st in SITES + ["Pooled"]:
                mk = np.ones(len(y), bool) if st == "Pooled" else site == st
                a_, b_, ici, oe = calib(y[mk], P[m][mk])
                cal.append({"set": "main", "seed": s, "method": m, "site": st, "intercept": a_, "slope": b_, "ici": ici, "o_e": oe})
    pairs = [(a, b) for a in REF for b in COMP + ["FedMedian-MLP"]]
    boot += joint_bootstrap(sets, pairs, "main")

    # ---- ext3: FedLR, robust aggregators, calibration fixes (same partitions/splits as main)
    ext3 = []
    for s in SEEDS:
        f = RUNS / f"ext3_p0.85_s{s}_preds.npz"
        if not f.exists():
            continue
        subj, yy = test_subjects(d, F, s)
        P = np.load(f, allow_pickle=True)
        assert np.array_equal(P["y"], yy)
        ext3.append((P["y"], subj, {m: P[m] for m in P.files if m not in ("y", "site", "female")}))
        for m in ("BFE-PW", "BFE-PW-Robust", "BFE-PW-SiteInt", "FedLR", "A7-BFE-PW", "A7-BFE-PW-Robust"):
            if m not in P.files:
                continue
            for st in SITES + ["Pooled"]:
                mk = np.ones(len(yy), bool) if st == "Pooled" else P["site"] == st
                a_, b_, ici, oe = calib(P["y"][mk], P[m][mk])
                cal.append({"set": "ext3", "seed": s, "method": m, "site": st, "intercept": a_, "slope": b_, "ici": ici, "o_e": oe})
    if len(ext3) == 5:
        boot += joint_bootstrap(ext3, [("BFE-PW", "FedLR"), ("BFE-Stack", "FedLR")], "ext3")

    # ---- care-era partition
    yr = []
    for s in SEEDS:
        f = RUNS / f"year_p0.85_s{s}_preds.npz"
        if not f.exists():
            continue
        subj, yy = test_subjects(d, F, s, partition_kind="year")
        y, site, P = load_preds(f, REF + COMP)
        assert np.array_equal(y, yy)
        yr.append((y, subj, P))
    if len(yr) == 5:
        boot += joint_bootstrap(yr, [(a, b) for a in REF for b in COMP], "year")

    # ---- data-scarce sites
    for fr in ("0.05", "0.1", "0.2"):
        sm = []
        for s in SEEDS:
            f = RUNS / f"small{fr}_p0.85_s{s}_preds.npz"
            if not f.exists():
                continue
            subj, yy = test_subjects(d, F, s)   # test sets are not subsampled
            y, site, P = load_preds(f, REF + COMP)
            assert np.array_equal(y, yy)
            sm.append((y, subj, P))
        if len(sm) == 5:
            boot += joint_bootstrap(sm, [(a, b) for a in REF for b in COMP], f"small{fr}")

    bt = pd.DataFrame(boot)
    bt.to_csv(TAB / "rev_joint_bootstrap.csv", index=False)
    print(bt.round(4).to_string())
    cl = pd.DataFrame(cal)
    cl.to_csv(TAB / "rev_calibration_long.csv", index=False)
    cs = cl.groupby(["set", "method", "site"]).agg(**{f"{k}_mean": (k, "mean") for k in ("intercept", "slope", "ici", "o_e")},
                                                    **{f"{k}_sd": (k, "std") for k in ("intercept", "slope", "ici", "o_e")}).reset_index()
    cs.to_csv(TAB / "rev_calibration_summary.csv", index=False)
    summ = [x for x in (seed_summary(p, l) for p, l in (("small0.05_", "small0.05"), ("small0.1_", "small0.1"),
                                                          ("small0.2_", "small0.2"), ("year_", "year"), ("ext3_", "ext3")))
            if x is not None]
    if summ:
        pd.concat(summ).to_csv(TAB / "rev_seed_summary.csv", index=False)
    print("done")
