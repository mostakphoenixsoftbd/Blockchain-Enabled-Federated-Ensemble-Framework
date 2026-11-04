"""Shared data preparation, non-IID partitioning and evaluation utilities."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (RESULTS, FIGURES):
    _d.mkdir(exist_ok=True)

SITES = ["A", "B", "C", "D"]
SITE_NAMES = {
    "A": "Hospital A (geriatric referral)",
    "B": "Hospital B (renal/nephrology)",
    "C": "Hospital C (community, sparse labs)",
    "D": "Hospital D (general acute care)",
}
# Panels that Hospital C does not run within the first 24 h (structural missingness)
SPARSE_PANELS = ["lactate", "ph", "troponin_t", "albumin", "ast", "alt", "alp",
                 "bilirubin", "lymphocytes_pct", "neutrophils_pct", "pt", "inr", "ptt"]
SITE_C_MCAR = 0.30   # additional random loss of the remaining lab results at Hospital C

CATEGORICAL = ["gender", "admission_type", "admission_location", "insurance",
               "marital_status", "race_group", "language_english"]
DROP = ["subject_id", "hadm_id", "admittime", "label", "race", "language", "anchor_year_group"]


def _race_group(r: pd.Series) -> pd.Series:
    r = r.fillna("UNKNOWN").str.upper()
    out = pd.Series("OTHER", index=r.index)
    for key, val in [("WHITE", "WHITE"), ("BLACK", "BLACK"), ("HISPANIC", "HISPANIC"),
                     ("LATINO", "HISPANIC"), ("ASIAN", "ASIAN"), ("UNKNOWN", "UNKNOWN"),
                     ("UNABLE", "UNKNOWN"), ("DECLINED", "UNKNOWN")]:
        out[r.str.contains(key, regex=False)] = val
    return out


# Feature sets: "base" (primary analysis, hosp labs/history), "ext" (+ ICU module, prescriptions,
# microbiology orders; 01b_build_ext_features.py), "ext2" (ext + bedside vitals from chartevents).
FEATURE_SETS = {"base": [], "ext": ["ext_icu_rx_micro"], "ext2": ["ext_icu_rx_micro", "ext_chart"]}


def load_cohort(features: str = "base") -> pd.DataFrame:
    d = pd.read_parquet(DATA / "cohort.parquet")
    f64 = d.select_dtypes("float64").columns
    d[f64] = d[f64].astype(np.float32)
    d["race_group"] = _race_group(d["race"])
    d["language_english"] = (d["language"].fillna("").str.upper() == "ENGLISH").astype(str)
    d["n_lab_results"] = d["n_lab_results"].fillna(0)
    d = d.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    # extra blocks are joined after sorting, so rows (and hence partitions and splits) match "base"
    for name in FEATURE_SETS[features]:
        e = pd.read_parquet(DATA / f"{name}.parquet").set_index("hadm_id").astype(np.float32)
        d = d.join(e, on="hadm_id")
    return d


def subject_profile(d: pd.DataFrame) -> pd.DataFrame:
    """Patient-level attributes that drive clinical-skew assignment."""
    g = d.groupby("subject_id")
    prof = pd.DataFrame({
        "age_first": g["age"].first(),
        "renal": (g["creatinine_max"].max() >= 2.0) | (g["hx_ckd"].max() == 1),
    })
    return prof


def partition(d: pd.DataFrame, purity: float, seed: int) -> pd.Series:
    """Assign every patient (subject_id) to one hospital.

    With probability `purity` a patient follows the clinical-skew rule
    (age>=70 -> A; renal disease -> B; others -> C/D at 40/60); otherwise the
    patient is placed uniformly at random. purity=0 gives an IID split.
    All admissions of a patient share a hospital, preventing cross-site leakage.
    """
    rng = np.random.default_rng(seed)
    prof = subject_profile(d)
    n = len(prof)
    rule = np.where(prof["age_first"] >= 70, "A",
                    np.where(prof["renal"], "B",
                             np.where(rng.random(n) < 0.4, "C", "D")))
    random_site = rng.choice(SITES, size=n)
    follow = rng.random(n) < purity
    site = pd.Series(np.where(follow, rule, random_site), index=prof.index, name="site")
    return d["subject_id"].map(site)


def split_subjects(d: pd.DataFrame, site: pd.Series, seed: int,
                   frac=(0.7, 0.1, 0.2)) -> pd.Series:
    """Patient-level train/validation/test split inside each hospital."""
    rng = np.random.default_rng(seed + 1000)
    subj = d[["subject_id"]].assign(site=site.values).drop_duplicates("subject_id")
    u = rng.random(len(subj))
    part = np.where(u < frac[0], "train", np.where(u < frac[0] + frac[1], "val", "test"))
    m = pd.Series(part, index=subj["subject_id"].values)
    return d["subject_id"].map(m)


def apply_site_c_missingness(X: pd.DataFrame, is_c: np.ndarray, seed: int) -> pd.DataFrame:
    """Structural (panel absent) + random (MCAR) lab missingness at Hospital C."""
    X = X.copy()
    rng = np.random.default_rng(seed + 2000)
    lab_cols = [c for c in X.columns if c.endswith(("_min", "_max", "_mean"))]
    sparse_cols = [c for c in lab_cols if c.rsplit("_", 1)[0] in SPARSE_PANELS]
    other_labs = sorted({c.rsplit("_", 1)[0] for c in lab_cols} - set(SPARSE_PANELS))
    idx = np.where(is_c)[0]
    X.iloc[idx, [X.columns.get_loc(c) for c in sparse_cols]] = np.nan
    for lab in other_labs:
        drop = idx[rng.random(len(idx)) < SITE_C_MCAR]
        cols = [X.columns.get_loc(f"{lab}_{s}") for s in ("min", "max", "mean")]
        X.iloc[drop, cols] = np.nan
    return X


def build_features(d: pd.DataFrame, features: str = "base") -> pd.DataFrame:
    """One-hot encode categoricals with a fixed, globally agreed vocabulary.

    The vocabulary (category names only, no counts) is the kind of schema a
    consortium agrees on up front (cf. common data models), so no patient-level
    information is exchanged by using it. Cached to data/features.parquet
    (rows aligned with load_cohort()).
    """
    cache = DATA / ("features.parquet" if features == "base" else f"features_{features}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    X = d.drop(columns=[c for c in DROP if c in d.columns]).copy()
    for c in CATEGORICAL:
        X[c] = X[c].fillna("UNKNOWN").astype(str)
    X = pd.get_dummies(X, columns=CATEGORICAL, dtype=np.float32)
    X["days_since_last_disch"] = X["days_since_last_disch"].astype(float)
    X = X.astype(np.float32)
    X.to_parquet(cache)
    return X


YEAR_SITES = {"2008 - 2010": "A", "2011 - 2013": "B", "2014 - 2016": "C",
              "2017 - 2019": "D", "2020 - 2022": "D"}


def partition_by_year(d: pd.DataFrame) -> pd.Series:
    """Natural (non-simulated) partition: each 'site' is one MIMIC-IV anchor-year group
    (care era). anchor_year_group is patient-level, so all admissions of a patient share
    a site. The two most recent groups are merged so the four sites have similar sizes."""
    return d["anchor_year_group"].map(YEAR_SITES)


def subsample_sites(d: pd.DataFrame, site: np.ndarray, split: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Data-scarce regime: keep a random fraction `frac` of each hospital's patients in the
    training and validation splits (patient level); test sets are untouched so that
    evaluation precision is unchanged. Dropped rows get split label 'drop'."""
    rng = np.random.default_rng(seed + 3000)
    subj = pd.Series(d["subject_id"].values).drop_duplicates()
    keep = pd.Series(rng.random(len(subj)) < frac, index=subj.values)
    kept = d["subject_id"].map(keep).values
    split = split.copy()
    split[np.isin(split, ["train", "val"]) & ~kept] = "drop"
    return split


def prepare(purity: float, seed: int, d: pd.DataFrame | None = None, X_full: pd.DataFrame | None = None,
            partition_kind: str = "rule", frac: float = 1.0):
    """partition_kind: 'rule' (simulated clinical-skew hospitals, with Hospital C lab
    missingness) or 'year' (natural care-era sites, no added missingness).
    frac < 1 subsamples each site's training and validation patients (data-scarce regime)."""
    d = load_cohort() if d is None else d
    X = build_features(d) if X_full is None else X_full
    if partition_kind == "year":
        site = partition_by_year(d).values
    else:
        site = partition(d, purity, seed).values
    split = split_subjects(d, pd.Series(site), seed).values
    if frac < 1.0:
        split = subsample_sites(d, site, split, frac, seed)
    if partition_kind == "rule":
        X = apply_site_c_missingness(X, site == "C", seed)
    y = d["label"].values.astype(np.int8)
    return X, y, site, split


# ---------------------------------------------------------------- metrics

def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def threshold_from_counts(y_list, p_list, grid=np.linspace(0.005, 0.9, 400)):
    """Choose the F1-optimal threshold from confusion counts summed across sites.

    Each hospital shares only (TP, FP, FN) counts at a fixed grid, so the
    threshold is selected federatedly without exchanging predictions.
    """
    tp = np.zeros_like(grid); fp = np.zeros_like(grid); fn = np.zeros_like(grid)
    for y, p in zip(y_list, p_list):
        pred = p[None, :] >= grid[:, None]
        tp += (pred & (y[None, :] == 1)).sum(1)
        fp += (pred & (y[None, :] == 0)).sum(1)
        fn += (~pred & (y[None, :] == 1)).sum(1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(grid[np.argmax(f1)])


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece(y, p)),
        "threshold": thr,
        "n": int(len(y)), "events": int(y.sum()),
    }


def delong_test(y, p1, p2):
    """Two-sided DeLong test for paired AUROCs (fast midrank implementation)."""
    from scipy.stats import norm, rankdata
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def comps(p):
        tx, ty, tz = rankdata(p[pos]), rankdata(p[neg]), rankdata(np.r_[p[pos], p[neg]])
        v10 = (tz[:m] - tx) / n
        v01 = 1.0 - (tz[m:] - ty) / m
        return tz[:m].sum() / (m * n) - (m + 1) / (2 * n), v10, v01

    a1, v10a, v01a = comps(p1)
    a2, v10b, v01b = comps(p2)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    s = s10 / m + s01 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return float(a1), float(a2), float(z), float(2 * norm.sf(abs(z)))


def bootstrap_ci(y, p, fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
