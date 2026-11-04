"""Model training: local tree ensembles, MLP (centralised / FedAvg / FedProx),
and the blockchain-audited federated ensemble (BFE)."""
import io
import json
import pickle
import time

import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from common import SITES, threshold_from_counts

torch.set_num_threads(2)   # small MLP: more threads only add contention

XGB_PARAMS = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                  eval_metric="aucpr", early_stopping_rounds=50, n_jobs=8)
RF_PARAMS = dict(n_estimators=200, max_depth=14, min_samples_leaf=20, max_features="sqrt",
                 n_jobs=8)
MLP_HIDDEN = (128, 64)
FED_ROUNDS = 40
LOCAL_EPOCHS = 1
BATCH = 256
LR = 1e-3
PROX_MU = 0.01
GATE_AUROC = 0.70          # pre-specified admission threshold (median peer AUROC)


# ------------------------------------------------------------------ trees

def train_xgb(Xtr, ytr, Xva, yva, seed):
    m = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def train_rf(Xtr, ytr, seed):
    return RandomForestClassifier(**RF_PARAMS, random_state=seed).fit(Xtr, ytr)


def serialize(model) -> bytes:
    if isinstance(model, xgb.XGBClassifier):
        return bytes(model.get_booster().save_raw(raw_format="ubj"))
    if isinstance(model, torch.nn.Module):
        buf = io.BytesIO(); torch.save(model.state_dict(), buf); return buf.getvalue()
    return pickle.dumps(model, protocol=5)


def predict(model, X):
    if isinstance(model, torch.nn.Module):
        raise TypeError("use MLPWrapper")
    return model.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ MLP

class Preproc:
    """Federated standardisation: sites share only per-column sums, sums of
    squares and non-missing counts of their training data."""

    def __init__(self, columns):
        self.cols = list(columns)
        self.logcols = np.array([c.endswith(("_min", "_max", "_mean", "_cnt", "_ml", "_hrs")) or
                                 c in ("n_prior_adm", "days_since_last_disch", "n_lab_results")
                                 for c in self.cols])

    def _tx(self, X):
        A = np.asarray(X, dtype=np.float64).copy()
        A[:, self.logcols] = np.log1p(np.clip(A[:, self.logcols], 0, None))
        return A

    def fit_federated(self, X_by_site):
        s = 0; ss = 0; n = 0
        for X in X_by_site:
            A = self._tx(X)
            s = s + np.nansum(A, 0); ss = ss + np.nansum(A ** 2, 0); n = n + (~np.isnan(A)).sum(0)
        self.mean = s / np.maximum(n, 1)
        self.std = np.sqrt(np.maximum(ss / np.maximum(n, 1) - self.mean ** 2, 1e-8))
        self.miss_cols = np.where(n < sum(len(X) for X in X_by_site))[0]
        return self

    def transform(self, X):
        A = self._tx(X)
        miss = np.isnan(A[:, self.miss_cols]).astype(np.float32)
        Z = np.clip((A - self.mean) / self.std, -5, 5)
        Z = np.nan_to_num(Z, nan=0.0)
        return np.hstack([Z, miss]).astype(np.float32)


def make_mlp(d_in, seed):
    torch.manual_seed(seed)
    h1, h2 = MLP_HIDDEN
    return torch.nn.Sequential(torch.nn.Linear(d_in, h1), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(h1, h2), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(h2, 1))


def _epochs(model, Z, y, epochs, seed, global_params=None, mu=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    Zt, yt = torch.from_numpy(Z), torch.from_numpy(y.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Zt), generator=g)
        for i in range(0, len(Zt), BATCH):
            b = perm[i:i + BATCH]
            loss = lossf(model(Zt[b]).squeeze(1), yt[b])
            if mu > 0:
                loss = loss + mu / 2 * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), global_params))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def mlp_predict(model, Z):
    model.eval()
    out = []
    for i in range(0, len(Z), 8192):
        out.append(torch.sigmoid(model(torch.from_numpy(Z[i:i + 8192])).squeeze(1)).numpy())
    return np.concatenate(out)


@torch.no_grad()
def _bce(model, Z, y):
    p = np.clip(mlp_predict(model, Z), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def train_mlp_central(Ztr, ytr, Zva, yva, seed, max_epochs=FED_ROUNDS):
    model = make_mlp(Ztr.shape[1], seed)
    best, best_state, patience = np.inf, None, 0
    for ep in range(max_epochs):
        _epochs(model, Ztr, ytr, 1, seed + ep)
        v = _bce(model, Zva, yva)
        if v < best:
            best, best_state, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state)
    return model


def _multi_krum(client_states, n, f=1):
    """Multi-Krum (Blanchard et al. 2017): score each client by the summed squared distance
    to its n_clients - f - 2 nearest neighbours, keep the n_clients - f lowest-scoring
    clients and average them weighted by sample size."""
    keys = [k for k, t in client_states[0].items() if t.is_floating_point()]
    vec = torch.stack([torch.cat([c[k].flatten().float() for k in keys]) for c in client_states])
    D = torch.cdist(vec, vec) ** 2
    k_nn = max(len(client_states) - f - 2, 1)
    scores = torch.stack([torch.sort(D[i])[0][1:k_nn + 1].sum() for i in range(len(client_states))])
    keep = torch.argsort(scores)[: len(client_states) - f].tolist()
    wts = np.array([n[i] for i in keep], dtype=float); wts /= wts.sum()
    out = {}
    for k, t in client_states[0].items():
        out[k] = sum(client_states[i][k] * w for i, w in zip(keep, wts)) if t.is_floating_point() else t
    return out, keep


def train_fed_mlp(site_data, seed, mu=0.0, rounds=FED_ROUNDS, attacker=None, agg="mean", y_override=None):
    """FedAvg (mu=0) or FedProx (mu>0). site_data: {site: (Ztr, ytr, Zva, yva)}.
    agg="median": Byzantine-robust coordinate-wise median of client weights (unweighted).
    agg="trimmed": coordinate-wise trimmed mean dropping the largest and smallest value (unweighted).
    agg="krum": Multi-Krum with f=1 (see _multi_krum).
    Model selection uses the federated validation loss (sum of site losses).
    `attacker`: site whose training labels are flipped (label-flipping poisoning).
    `y_override`: {site: poisoned training labels} for partial or targeted poisoning."""
    d_in = next(iter(site_data.values()))[0].shape[1]
    gmodel = make_mlp(d_in, seed)
    n = {s: len(v[1]) for s, v in site_data.items()}
    N = sum(n.values())
    best, best_state, history = np.inf, None, []
    for r in range(rounds):
        gstate = {k: t.clone() for k, t in gmodel.state_dict().items()}
        gparams = [p.detach().clone() for p in gmodel.parameters()]
        new = {k: torch.zeros_like(t) for k, t in gstate.items()}
        client_states = []
        for i, (s, (Ztr, ytr, _, _)) in enumerate(site_data.items()):
            local = make_mlp(d_in, seed); local.load_state_dict(gstate)
            y_use = 1 - ytr if s == attacker else ytr
            if y_override is not None and s in y_override:
                y_use = y_override[s]
            _epochs(local, Ztr, y_use, LOCAL_EPOCHS, seed * 1000 + r * 10 + i, gparams, mu)
            client_states.append({k: t.clone() for k, t in local.state_dict().items()})
            for k, t in client_states[-1].items():
                new[k] += t * (n[s] / N)
        if agg == "median":
            new = {k: torch.quantile(torch.stack([c[k].float() for c in client_states]), 0.5, dim=0).to(gstate[k].dtype)
                   if gstate[k].is_floating_point() else new[k] for k in gstate}
        elif agg == "trimmed":
            new = {k: torch.sort(torch.stack([c[k].float() for c in client_states]), dim=0)[0][1:-1].mean(0).to(gstate[k].dtype)
                   if gstate[k].is_floating_point() else new[k] for k in gstate}
        elif agg == "krum":
            new, _ = _multi_krum(client_states, [n[s] for s in site_data])
        gmodel.load_state_dict(new)
        vloss = sum(_bce(gmodel, v[2], v[3]) for v in site_data.values())
        history.append(vloss)
        if vloss < best:
            best, best_state = vloss, {k: t.clone() for k, t in gmodel.state_dict().items()}
    gmodel.load_state_dict(best_state)
    return gmodel, history


# ------------------------------------------------------------------ federated logistic combiner

def fed_logistic(F_by_site, y_by_site, l2=1e-3, iters=25):
    """Federated L2-logistic regression by Newton-Raphson: each iteration every
    site returns only the gradient and Hessian of its log-likelihood with respect
    to the (k+1) shared coefficients, never patient-level rows."""
    k = F_by_site[0].shape[1]
    w = np.zeros(k + 1)
    N = sum(len(y) for y in y_by_site)
    reg = np.r_[np.full(k, l2), 0.0]
    for _ in range(iters):
        g = np.zeros_like(w); H = np.zeros((k + 1, k + 1))
        for F, y in zip(F_by_site, y_by_site):
            A = np.hstack([F, np.ones((len(F), 1))])
            p = 1 / (1 + np.exp(-(A @ w)))
            g += A.T @ (p - y)
            H += (A * (p * (1 - p))[:, None]).T @ A
        g = g / N + reg * w
        H = H / N + np.diag(reg) + 1e-9 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ BFE protocol

class BFE:
    """Blockchain-audited federated ensemble.

    members: list of dicts {owner, family, model, bytes}. Peer attestations and the
    admission decision live on-chain; aggregation weights are a deterministic
    function of on-chain attestations, so any participant can recompute them.
    """

    def __init__(self, chain, site_nodes, round_base, gate=GATE_AUROC, malicious_validators=()):
        self.ch, self.nodes, self.rb, self.gate = chain, site_nodes, round_base, gate
        self.malicious_validators = set(malicious_validators)
        self.timing = {}

    def run(self, members, X_val, y_val, n_train):
        t0 = time.perf_counter()
        families = sorted({m["family"] for m in members})
        rid = {f: self.rb + i for i, f in enumerate(families)}
        for f in families:
            self.ch.open_round(rid[f], self.gate)
        for m in members:
            meta = json.dumps({"family": m["family"], "site": m["owner"]}).encode()
            self.ch.submit(self.nodes[m["owner"]], rid[m["family"]], m["bytes"], meta, n_train[m["owner"]])
        t_eval = 0.0
        # off-chain: every peer verifies the received artefact against the on-chain digest,
        # evaluates it on its own validation split and posts the score on-chain
        for m in members:
            for v in SITES:
                if v == m["owner"]:
                    continue
                te = time.perf_counter()
                if v in self.malicious_validators:
                    score = 0.0          # adversarial validator smears every peer
                else:
                    score = roc_auc_score(y_val[v], m["predict"](X_val[v]))
                t_eval += time.perf_counter() - te
                self.ch.attest(self.nodes[v], rid[m["family"]], self.nodes[m["owner"]], m["bytes"],
                               score, f"{v}->{m['owner']}:{m['family']}:{score:.4f}".encode())
        for f in families:
            self.ch.finalize(rid[f])
        for m in members:
            s = self.ch.submission(rid[m["family"]], self.nodes[m["owner"]])
            m["accepted"], m["median_auroc"] = s["accepted"], s["median_auroc"]
        self.members = members
        self.accepted = [m for m in members if m["accepted"]]
        w = np.array([max(m["median_auroc"] - 0.5, 0.0) for m in self.accepted])
        self.w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / max(len(w), 1)
        self.rid = rid
        self.timing["protocol_s"] = time.perf_counter() - t0 - t_eval
        self.timing["peer_eval_s"] = t_eval
        return self

    def member_logits(self, X):
        return np.column_stack([logit(m["predict"](X)) for m in self.accepted])

    def fit_combiners(self, X_val, y_val):
        F = {s: self.member_logits(X_val[s]) for s in SITES}
        # (i) performance-weighted logit pooling + federated Platt recalibration
        z = [(F[s] @ self.w)[:, None] for s in SITES]
        self.platt = fed_logistic(z, [y_val[s] for s in SITES], l2=0.0)
        # (ii) federated stacking over admitted members
        self.stack = fed_logistic([F[s] for s in SITES], [y_val[s] for s in SITES], l2=1e-3)
        blob = json.dumps({"w": self.w.tolist(), "platt": self.platt.tolist(),
                           "stack": self.stack.tolist(),
                           "members": [[m["owner"], m["family"]] for m in self.accepted]}).encode()
        self.ch.commit_global(min(self.rid.values()), blob, blob)
        return self

    def fit_platt_robust(self, X_val, y_val):
        """Robust recalibration: every site fits its own Platt (slope, intercept) on its
        validation split and shares only those two numbers; the consortium takes the
        coordinate-wise median, so one site cannot move the calibration on its own."""
        per_site = []
        for s in SITES:
            z = (self.member_logits(X_val[s]) @ self.w)[:, None]
            per_site.append(fed_logistic([z], [y_val[s]], l2=0.0))
        self.platt_sites = np.array(per_site)
        self.platt_robust = np.median(self.platt_sites, axis=0)
        return self

    def predict_pw_robust(self, X):
        z = self.member_logits(X) @ self.w
        return sigmoid(self.platt_robust[0] * z + self.platt_robust[1])

    def predict_pw(self, X):
        z = self.member_logits(X) @ self.w
        return sigmoid(self.platt[0] * z + self.platt[1])

    def predict_stack(self, X):
        F = self.member_logits(X)
        return sigmoid(F @ self.stack[:-1] + self.stack[-1])


def site_intercept(p_val, y_val, iters=50):
    """Local recalibration-in-the-large: the intercept shift a that makes
    sigmoid(logit(p) + a) fit the site's own validation outcomes (slope fixed at 1).
    Uses only the site's own data; nothing is shared."""
    z = logit(p_val); a = 0.0
    for _ in range(iters):
        q = sigmoid(z + a)
        g = (q - y_val).sum(); h = (q * (1 - q)).sum() + 1e-9
        step = g / h; a -= step
        if abs(step) < 1e-10:
            break
    return float(a)


def fed_threshold(predict_fn, X_val, y_val):
    return threshold_from_counts([y_val[s] for s in SITES], [predict_fn(X_val[s]) for s in SITES])
