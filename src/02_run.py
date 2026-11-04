"""Run all methods for one (purity, seed) configuration.

usage: python 02_run.py --purity 0.85 --seed 0 [--attacks] [--features ext]
Writes results/runs/p{purity}_s{seed}.json and pooled-test predictions (.npz);
non-base feature sets are prefixed, e.g. ext_p0.85_s0.json.
"""
import argparse
import json
import time
import warnings
from collections import defaultdict

import numpy as np

from chain import AuditChain
from common import RESULTS, SITES, load_cohort, build_features, metrics, prepare, threshold_from_counts
from fedlib import (BFE, Preproc, fed_threshold, mlp_predict, serialize, train_fed_mlp,
                    train_mlp_central, train_rf, train_xgb, PROX_MU)

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--purity", type=float, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--attacks", action="store_true")
ap.add_argument("--features", default="base", choices=["base", "ext", "ext2"])
ap.add_argument("--partition", default="rule", choices=["rule", "year"],
                help="rule: simulated clinical-skew hospitals; year: natural care-era sites")
ap.add_argument("--frac", type=float, default=1.0,
                help="fraction of each site's train/val patients kept (data-scarce regime)")
args = ap.parse_args()
P, SEED = args.purity, args.seed
OUT = RESULTS / "runs"; OUT.mkdir(exist_ok=True)
tag = ("" if args.features == "base" else f"{args.features}_") \
    + ("year_" if args.partition == "year" else "") \
    + (f"small{args.frac:g}_" if args.frac < 1 else "") + f"p{P:.2f}_s{SEED}"
T = {}
t_all = time.perf_counter()


def log(msg):
    print(f"[{tag} {time.perf_counter() - t_all:7.0f}s] {msg}", flush=True)

d = load_cohort(args.features)
X, y, site, split = prepare(P, SEED, d, build_features(d, args.features),
                            partition_kind=args.partition, frac=args.frac)
del d
cols = X.columns
Xv = X.values
I = {(s, sp): np.where((site == s) & (split == sp))[0] for s in SITES for sp in ("train", "val", "test")}
Xs = {k: Xv[i] for k, i in I.items()}
ys = {k: y[i] for k, i in I.items()}
Xval = {s: Xs[s, "val"] for s in SITES}; yval = {s: ys[s, "val"] for s in SITES}
test_idx = np.concatenate([I[s, "test"] for s in SITES])
test_site = site[test_idx]
yt = y[test_idx]
n_train = {s: len(ys[s, "train"]) for s in SITES}

res = {"purity": P, "seed": SEED, "features": args.features, "partition": args.partition, "frac": args.frac, "sites": {}, "methods": {}, "per_site": defaultdict(dict),
       "timing": T, "chain": {}, "cross_site_auroc": {}}
for s in SITES:
    res["sites"][s] = {sp: {"n": int(len(ys[s, sp])), "events": int(ys[s, sp].sum())} for sp in ("train", "val", "test")}
preds = {}


def record(name, p_test, thr, own_thr=None):
    preds[name] = p_test.astype(np.float32)
    res["methods"][name] = metrics(yt, p_test, thr)
    for s in SITES:
        m = test_site == s
        res["per_site"][name][s] = metrics(yt[m], p_test[m], own_thr[s] if own_thr else thr)


def pooled(fn):
    """Score the pooled test set with a per-row function of the site's raw features."""
    return fn(Xv[test_idx])


# ------------------------------------------------------------ local models
t = time.perf_counter()
local = {}
for s in SITES:
    tx = time.perf_counter()
    local[s, "xgb"] = train_xgb(Xs[s, "train"], ys[s, "train"], Xval[s], yval[s], SEED)
    T[f"local_xgb_{s}_s"] = time.perf_counter() - tx
    tx = time.perf_counter()
    local[s, "rf"] = train_rf(Xs[s, "train"], ys[s, "train"], SEED)
    T[f"local_rf_{s}_s"] = time.perf_counter() - tx
T["local_trees_total_s"] = time.perf_counter() - t
log("local trees trained")

# local-only deployment: each hospital scores its own patients with its own XGBoost model
p_local = np.zeros(len(test_idx)); thr_local = {}
for s in SITES:
    m = test_site == s
    p_local[m] = local[s, "xgb"].predict_proba(Xv[test_idx[m]])[:, 1]
    thr_local[s] = threshold_from_counts([yval[s]], [local[s, "xgb"].predict_proba(Xval[s])[:, 1]])
preds["Local-XGB"] = p_local.astype(np.float32)
res["methods"]["Local-XGB"] = metrics(yt, p_local, float(np.mean(list(thr_local.values()))))
for s in SITES:
    m = test_site == s
    res["per_site"]["Local-XGB"][s] = metrics(yt[m], p_local[m], thr_local[s])
for a in SITES:
    for b in SITES:
        m = test_site == b
        res["cross_site_auroc"][f"{a}->{b}"] = float(metrics(yt[m], local[a, "xgb"].predict_proba(Xv[test_idx[m]])[:, 1], 0.5)["auroc"])

# ------------------------------------------------------------ centralised (pooled data; privacy-violating upper bound)
Xtr_all = np.concatenate([Xs[s, "train"] for s in SITES]); ytr_all = np.concatenate([ys[s, "train"] for s in SITES])
Xva_all = np.concatenate([Xval[s] for s in SITES]); yva_all = np.concatenate([yval[s] for s in SITES])
t = time.perf_counter()
cx = train_xgb(Xtr_all, ytr_all, Xva_all, yva_all, SEED); T["central_xgb_s"] = time.perf_counter() - t
log("central xgb trained")
record("Central-XGB", pooled(lambda A: cx.predict_proba(A)[:, 1]),
       fed_threshold(lambda A: cx.predict_proba(A)[:, 1], Xval, yval))
t = time.perf_counter()
crf = train_rf(Xtr_all, ytr_all, SEED); T["central_rf_s"] = time.perf_counter() - t
record("Central-RF", pooled(lambda A: crf.predict_proba(A)[:, 1]),
       fed_threshold(lambda A: crf.predict_proba(A)[:, 1], Xval, yval))

# ------------------------------------------------------------ MLP family
pre = Preproc(cols).fit_federated([Xs[s, "train"] for s in SITES])
Z = {k: pre.transform(v) for k, v in Xs.items()}
Zval = {s: Z[s, "val"] for s in SITES}
Ztest = pre.transform(Xv[test_idx])
t = time.perf_counter()
cm = train_mlp_central(np.concatenate([Z[s, "train"] for s in SITES]), ytr_all,
                       np.concatenate([Zval[s] for s in SITES]), yva_all, SEED)
T["central_mlp_s"] = time.perf_counter() - t
log("central mlp trained")
record("Central-MLP", mlp_predict(cm, Ztest), fed_threshold(lambda A: mlp_predict(cm, A), Zval, yval))

site_data = {s: (Z[s, "train"], ys[s, "train"], Zval[s], yval[s]) for s in SITES}
fed_models = {}
for name, mu in (("FedAvg-MLP", 0.0), ("FedProx-MLP", PROX_MU)):
    t = time.perf_counter()
    fm, hist = train_fed_mlp(site_data, SEED, mu=mu)
    T[f"{name}_s"] = time.perf_counter() - t
    log(f"{name} trained")
    fed_models[name] = fm
    res[f"{name}_val_loss_curve"] = hist
    record(name, mlp_predict(fm, Ztest), fed_threshold(lambda A: mlp_predict(fm, A), Zval, yval))

local_mlp = {}
t = time.perf_counter()
for s in SITES:
    local_mlp[s] = train_mlp_central(Z[s, "train"], ys[s, "train"], Zval[s], yval[s], SEED)
T["local_mlp_total_s"] = time.perf_counter() - t
log("local mlps trained")


# ------------------------------------------------------------ ensembles
def tree_members(models):
    out = []
    for (s, fam), mdl in models.items():
        out.append({"owner": s, "family": fam, "model": mdl, "bytes": serialize(mdl),
                    "predict": (lambda A, mdl=mdl: mdl.predict_proba(A)[:, 1])})
    return out


def run_bfe(members, label, chain, round_base, malicious_validators=(), Xv_=None, Xte=None):
    Xv_ = Xval if Xv_ is None else Xv_
    Xte = Xv[test_idx] if Xte is None else Xte
    b = BFE(chain, dict(zip(SITES, chain.nodes)), round_base,
            malicious_validators=malicious_validators).run(members, Xv_, yval, n_train)
    res["chain"][label] = {
        "members": [{"owner": m["owner"], "family": m["family"], "accepted": bool(m["accepted"]),
                     "median_peer_auroc": m["median_auroc"], "bytes": len(m["bytes"])} for m in b.members],
        "weights": b.w.tolist(), "timing": b.timing, "global_model": bool(b.accepted),
    }
    if not b.accepted:
        # every member rejected: the contract withholds a global model (fail-closed)
        log(f"{label}: no member admitted -> no global model")
        return b
    b.fit_combiners(Xv_, yval)
    record(f"{label}-PW", b.predict_pw(Xte), fed_threshold(b.predict_pw, Xv_, yval))
    record(f"{label}-Stack", b.predict_stack(Xte), fed_threshold(b.predict_stack, Xv_, yval))
    res["chain"][label].update(stack_coef=b.stack.tolist(), platt=b.platt.tolist())
    log(f"{label}: done")
    return b


def uniform(members, label, Xv_=None, Xte=None):
    Xv_ = Xval if Xv_ is None else Xv_
    Xte = Xv[test_idx] if Xte is None else Xte
    f = lambda A: np.mean([m["predict"](A) for m in members], axis=0)
    record(label, f(Xte), fed_threshold(f, Xv_, yval))


chain = AuditChain(len(SITES), extra_accounts=2)
for s, node in zip(SITES, chain.nodes):
    chain.register(node, f"hospital-{s}")

honest = tree_members({k: v for k, v in local.items()})
uniform(honest, "Ens-Uniform")
t = time.perf_counter()
run_bfe(honest, "BFE", chain, 100)
T["bfe_protocol_total_s"] = time.perf_counter() - t

mlp_members = [{"owner": s, "family": "mlp", "model": local_mlp[s], "bytes": serialize(local_mlp[s]),
                "predict": (lambda A, m=local_mlp[s]: mlp_predict(m, A))} for s in SITES]
run_bfe(mlp_members, "BFE-MLPmembers", chain, 200, Xv_=Zval, Xte=Ztest)

# ------------------------------------------------------------ adversarial scenarios
if args.attacks:
    log("attack scenarios")
    ATT = "D"
    flipped = {
        (ATT, "xgb"): train_xgb(Xs[ATT, "train"], 1 - ys[ATT, "train"], Xval[ATT], 1 - yval[ATT], SEED),
        (ATT, "rf"): train_rf(Xs[ATT, "train"], 1 - ys[ATT, "train"], SEED),
    }
    poisoned = tree_members({**{k: v for k, v in local.items() if k[0] != ATT}, **flipped})
    # A1: label-flipping hospital
    uniform(poisoned, "A1-Ens-Uniform")
    run_bfe(poisoned, "A1-BFE", chain, 300)
    fm, _ = train_fed_mlp(site_data, SEED, mu=0.0, attacker=ATT)
    record("A1-FedAvg-MLP", mlp_predict(fm, Ztest), fed_threshold(lambda A: mlp_predict(fm, A), Zval, yval))
    # A2: label-flipping hospital that also submits adversarial peer attestations
    run_bfe(poisoned, "A2-BFE", chain, 400, malicious_validators=(ATT,))
    # A3: artefact substitution - D commits an honest model, then ships the flipped one
    sub = [dict(m) for m in honest]
    swapped = 0
    for m in sub:
        if m["owner"] == ATT:
            fm_ = flipped[ATT, m["family"]]
            m["model"], m["predict"] = fm_, (lambda A, mdl=fm_: mdl.predict_proba(A)[:, 1])
            m["delivered_bytes"] = serialize(fm_)
            swapped += 1
    # without an integrity check the aggregator ensembles whatever it receives
    uniform(sub, "A3-Ens-Uniform-noverify")
    # with the ledger: peers evaluated the honest artefact (hash-bound attestations); at aggregation the
    # aggregator re-hashes the delivered artefact and calls verifyModel(); mismatches are excluded
    b = BFE(chain, dict(zip(SITES, chain.nodes)), 500).run(
        [dict(m, predict=(lambda A, mdl=local[m["owner"], m["family"]]: mdl.predict_proba(A)[:, 1])) for m in sub],
        Xval, yval, n_train)
    detected = 0
    for m_run, m_sub in zip(b.members, sub):
        delivered = m_sub.get("delivered_bytes", m_sub["bytes"])
        ok = chain.verify(b.rid[m_run["family"]], chain.nodes[SITES.index(m_run["owner"])], delivered)
        if not ok and m_run["accepted"]:
            detected += 1
        m_run["accepted"] = m_run["accepted"] and ok
        m_run["predict"] = m_sub["predict"]
    b.accepted = [m for m in b.members if m["accepted"]]
    w = np.array([max(m["median_auroc"] - 0.5, 0) for m in b.accepted]); b.w = w / w.sum()
    b.fit_combiners(Xval, yval)
    record("A3-BFE-PW", b.predict_pw(Xv[test_idx]), fed_threshold(b.predict_pw, Xval, yval))
    # the same weighted ensemble without hash verification (substituted models kept)
    b2 = BFE(chain, dict(zip(SITES, chain.nodes)), 600).run(
        [dict(m, predict=(lambda A, mdl=local[m["owner"], m["family"]]: mdl.predict_proba(A)[:, 1])) for m in sub],
        Xval, yval, n_train)
    for m_run, m_sub in zip(b2.members, sub):
        m_run["predict"] = m_sub["predict"]
    b2.accepted = [m for m in b2.members if m["accepted"]]
    b2.fit_combiners(Xval, yval)
    record("A3-PW-noverify", b2.predict_pw(Xv[test_idx]), fed_threshold(b2.predict_pw, Xval, yval))
    res["attack"] = {"attacker": ATT, "A3_swapped": swapped, "A3_detected": detected}
    # A4: two colluding hospitals (C, D) flip labels and smear peers -> honest-majority assumption broken
    fl2 = dict(flipped)
    fl2[("C", "xgb")] = train_xgb(Xs["C", "train"], 1 - ys["C", "train"], Xval["C"], 1 - yval["C"], SEED)
    fl2[("C", "rf")] = train_rf(Xs["C", "train"], 1 - ys["C", "train"], SEED)
    pois2 = tree_members({**{k: v for k, v in local.items() if k[0] not in ("C", ATT)}, **fl2})
    run_bfe(pois2, "A4-BFE", chain, 700, malicious_validators=("C", ATT))
    uniform(pois2, "A4-Ens-Uniform")

# ------------------------------------------------------------ chain accounting
ops = defaultdict(list)
for op, gas, lat, ok in chain.log:
    if ok:
        ops[op].append((gas, lat))
res["chain"]["ops"] = {op: {"count": len(v), "gas_mean": float(np.mean([g for g, _ in v])),
                            "latency_ms_mean": float(1000 * np.mean([l for _, l in v]))} for op, v in ops.items()}
res["chain"]["rejected_tx"] = sum(1 for *_, ok in chain.log if not ok)
res["chain"]["bytecode_bytes"] = chain.bytecode_size
res["model_bytes"] = {"xgb_mean": float(np.mean([len(serialize(local[s, "xgb"])) for s in SITES])),
                      "rf_mean": float(np.mean([len(serialize(local[s, "rf"])) for s in SITES])),
                      "mlp": len(serialize(cm))}
T["total_s"] = time.perf_counter() - t_all
res["per_site"] = dict(res["per_site"])
(OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / f"{tag}_preds.npz", y=yt, site=test_site, **preds)
print(f"[{tag}] done in {T['total_s']:.0f}s")
for k, v in res["methods"].items():
    print(f"  {k:24s} AUROC {v['auroc']:.4f}  AUPRC {v['auprc']:.4f}  F1 {v['f1']:.4f}  Acc {v['accuracy']:.4f}  Brier {v['brier']:.4f}")
