"""Revision analyses (2026-11) on the main configuration (simulated hospitals, purity 0.85).

usage: python 02c_extensions.py --seed 0
Writes results/runs/ext3_p0.85_s{seed}.json and _preds.npz (same pooled-test order as 02_run.py).

B2  stealthier poisoning by Hospital D:
    A5q  partial label flip (a random fraction q of D's training and validation labels flipped)
    A6   targeted flip: deaths among female patients at D relabelled as survivals
    A7   honest models, but D poisons the federated recalibration and threshold
         (submits flipped validation labels to those steps)
    each against BFE, an oracle ensemble that excludes D by hand, and FedAvg-MLP;
    A1 (full flip) and clean training also against trimmed-mean and Multi-Krum FedAvg.
B3  calibration fixes: robust (median-of-sites) Platt recalibration and a local
    site-specific intercept update (each hospital uses only its own validation data).
B5  comparator: federated logistic regression on the shared feature space.
"""
import argparse
import json
import time
import warnings

import numpy as np

from chain import AuditChain
from common import RESULTS, SITES, build_features, load_cohort, metrics, prepare, threshold_from_counts
from fedlib import (BFE, Preproc, fed_logistic, fed_threshold, logit, mlp_predict, serialize, sigmoid,
                    site_intercept, train_fed_mlp, train_rf, train_xgb)

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--purity", type=float, default=0.85)
args = ap.parse_args()
P, SEED = args.purity, args.seed
OUT = RESULTS / "runs"
tag = f"ext3_p{P:.2f}_s{SEED}"
t_all = time.perf_counter()
ATT = "D"
FLIP_FRACS = (0.2, 0.4)


def log(msg):
    print(f"[{tag} {time.perf_counter() - t_all:7.0f}s] {msg}", flush=True)


d = load_cohort()
X, y, site, split = prepare(P, SEED, d, build_features(d))
del d
cols, Xv = X.columns, X.values
female = (X["gender_F"].values == 1) if "gender_F" in cols else None
I = {(s, sp): np.where((site == s) & (split == sp))[0] for s in SITES for sp in ("train", "val", "test")}
Xs = {k: Xv[i] for k, i in I.items()}
ys = {k: y[i] for k, i in I.items()}
Xval = {s: Xs[s, "val"] for s in SITES}; yval = {s: ys[s, "val"] for s in SITES}
test_idx = np.concatenate([I[s, "test"] for s in SITES])
test_site, yt = site[test_idx], y[test_idx]
Xte = Xv[test_idx]
fem_te = female[test_idx]
n_train = {s: len(ys[s, "train"]) for s in SITES}

res = {"purity": P, "seed": SEED, "methods": {}, "per_site": {}, "subgroup_sex": {}, "chain": {}, "params": {}}
preds = {}


def record(name, p_test, thr, site_thr=None):
    preds[name] = p_test.astype(np.float32)
    res["methods"][name] = metrics(yt, p_test, thr)
    res["per_site"][name] = {s: metrics(yt[test_site == s], p_test[test_site == s],
                                        site_thr[s] if site_thr else thr) for s in SITES}
    res["subgroup_sex"][name] = {g: metrics(yt[m], p_test[m], thr) for g, m in (("F", fem_te), ("M", ~fem_te))}
    m = res["methods"][name]
    log(f"{name:28s} AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} ECE {m['ece']:.4f}")


def tree_members(models):
    return [{"owner": s, "family": fam, "model": mdl, "bytes": serialize(mdl),
             "predict": (lambda A, mdl=mdl: mdl.predict_proba(A)[:, 1])} for (s, fam), mdl in models.items()]


chain = AuditChain(len(SITES), extra_accounts=2)
for s, node in zip(SITES, chain.nodes):
    chain.register(node, f"hospital-{s}")
RB = [1000]


def run_bfe(members, label, y_for_combiners=None, robust=False, site_int=False):
    """Run the on-chain protocol; combiners and threshold use y_for_combiners (poisoned
    labels in A7). Records PW, Stack and, optionally, the robust and site-intercept variants."""
    RB[0] += 100
    yc = yval if y_for_combiners is None else y_for_combiners
    b = BFE(chain, dict(zip(SITES, chain.nodes)), RB[0]).run(members, Xval, yval, n_train)
    res["chain"][label] = {"members": [{"owner": m["owner"], "family": m["family"], "accepted": bool(m["accepted"]),
                                        "median_peer_auroc": m["median_auroc"]} for m in b.members],
                           "weights": b.w.tolist(), "global_model": bool(b.accepted)}
    if not b.accepted:
        log(f"{label}: no member admitted")
        return b
    b.fit_combiners(Xval, yc)
    res["params"][label] = {"platt": b.platt.tolist(), "stack": b.stack.tolist()}
    record(f"{label}-PW", b.predict_pw(Xte), fed_threshold(b.predict_pw, Xval, yc))
    record(f"{label}-Stack", b.predict_stack(Xte), fed_threshold(b.predict_stack, Xval, yc))
    if robust:
        b.fit_platt_robust(Xval, yc)
        # robust threshold: median of the site-optimal thresholds (each site shares one number)
        thr = float(np.median([threshold_from_counts([yc[s]], [b.predict_pw_robust(Xval[s])]) for s in SITES]))
        res["params"][label].update(platt_sites=b.platt_sites.tolist(), platt_robust=b.platt_robust.tolist())
        record(f"{label}-PW-Robust", b.predict_pw_robust(Xte), thr)
    if site_int:
        # each hospital shifts the global model's intercept on its own validation data, then
        # uses its own F1-optimal threshold for its own patients
        p = b.predict_pw(Xte).copy(); a = {}; thr_s = {}
        for s in SITES:
            a[s] = site_intercept(b.predict_pw(Xval[s]), yval[s])
            p[test_site == s] = sigmoid(logit(p[test_site == s]) + a[s])
            thr_s[s] = threshold_from_counts([yval[s]], [sigmoid(logit(b.predict_pw(Xval[s])) + a[s])])
        res["params"][label]["site_intercepts"] = a
        record(f"{label}-PW-SiteInt", p, float(np.mean(list(thr_s.values()))), site_thr=thr_s)
    return b


# ------------------------------------------------------------ honest local members
local = {}
for s in SITES:
    local[s, "xgb"] = train_xgb(Xs[s, "train"], ys[s, "train"], Xval[s], yval[s], SEED)
    local[s, "rf"] = train_rf(Xs[s, "train"], ys[s, "train"], SEED)
log("local trees trained")
honest = tree_members(local)
run_bfe(honest, "BFE", robust=True, site_int=True)

# ------------------------------------------------------------ MLP comparators and federated LR
pre = Preproc(cols).fit_federated([Xs[s, "train"] for s in SITES])
Z = {k: pre.transform(v) for k, v in Xs.items()}
Zval = {s: Z[s, "val"] for s in SITES}
Ztest = pre.transform(Xte)
site_data = {s: (Z[s, "train"], ys[s, "train"], Zval[s], yval[s]) for s in SITES}

wlr = fed_logistic([Z[s, "train"] for s in SITES], [ys[s, "train"] for s in SITES], l2=1e-3)
flr = lambda A: sigmoid(A @ wlr[:-1] + wlr[-1])
record("FedLR", flr(Ztest), fed_threshold(flr, Zval, yval))


def fed_mlp(name, **kw):
    fm, _ = train_fed_mlp(site_data, SEED, **kw)
    record(name, mlp_predict(fm, Ztest), fed_threshold(lambda A: mlp_predict(fm, A), Zval, yval))


fed_mlp("FedAvg-MLP")
for agg, nm in (("trimmed", "FedTrim-MLP"), ("krum", "FedKrum-MLP")):
    fed_mlp(nm, agg=agg)
    fed_mlp(f"A1-{nm}", agg=agg, attacker=ATT)

# ------------------------------------------------------------ B2: stealthier poisoning by D
rng = np.random.default_rng(SEED + 4000)
oracle = tree_members({k: v for k, v in local.items() if k[0] != ATT})
run_bfe(oracle, "Oracle-noD")

poison_sets = {}
for q in FLIP_FRACS:
    ytr_p = ys[ATT, "train"].copy(); yva_p = yval[ATT].copy()
    ftr = rng.random(len(ytr_p)) < q; fva = rng.random(len(yva_p)) < q
    ytr_p[ftr] = 1 - ytr_p[ftr]; yva_p[fva] = 1 - yva_p[fva]
    poison_sets[f"A5q{int(q * 100)}"] = (ytr_p, yva_p)
fem_tr = female[I[ATT, "train"]]; fem_va = female[I[ATT, "val"]]
ytr_t = ys[ATT, "train"].copy(); ytr_t[fem_tr] = 0
yva_t = yval[ATT].copy(); yva_t[fem_va] = 0
poison_sets["A6"] = (ytr_t, yva_t)
res["poison_counts"] = {k: {"train_labels_changed": int((v[0] != ys[ATT, "train"]).sum()),
                            "val_labels_changed": int((v[1] != yval[ATT]).sum())} for k, v in poison_sets.items()}

for name, (ytr_p, yva_p) in poison_sets.items():
    pm = {(ATT, "xgb"): train_xgb(Xs[ATT, "train"], ytr_p, Xval[ATT], yva_p, SEED),
          (ATT, "rf"): train_rf(Xs[ATT, "train"], ytr_p, SEED)}
    members = tree_members({**{k: v for k, v in local.items() if k[0] != ATT}, **pm})
    run_bfe(members, f"{name}-BFE")
    fed_mlp(f"{name}-FedAvg-MLP", y_override={ATT: ytr_p})
    log(f"{name} done")

# A7: honest members, poisoned recalibration and threshold (D submits flipped validation labels)
y_pois = dict(yval); y_pois[ATT] = 1 - yval[ATT]
run_bfe(honest, "A7-BFE", y_for_combiners=y_pois, robust=True)

(OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / f"{tag}_preds.npz", y=yt, site=test_site, female=fem_te, **preds)
log("done")
