"""Gas, latency and scalability of the audit protocol, plus tamper-detection trials."""
import json
import time
import warnings
from collections import defaultdict

import numpy as np
import xgboost as xgb

from chain import AuditChain, sha256
from common import RESULTS

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
out = {"scaling": [], "tamper": {}}

for K in (4, 8, 16, 32):
    ch = AuditChain(K, extra_accounts=1)
    for i, n in enumerate(ch.nodes):
        ch.register(n, f"hospital-{i}")
    t0 = time.perf_counter()
    ch.open_round(1, 0.70)
    models = [rng.bytes(64) for _ in range(K)]
    for n, m in zip(ch.nodes, models):
        ch.submit(n, 1, m, b"meta", 50000)
    for v in ch.nodes:
        for j, o in enumerate(ch.nodes):
            if v != o:
                ch.attest(v, 1, o, models[j], float(rng.uniform(0.75, 0.9)), b"eval")
    ch.finalize(1)
    ch.commit_global(1, b"global", b"weights")
    wall = time.perf_counter() - t0
    ops = defaultdict(list)
    for op, gas, lat, ok in ch.log:
        if ok and op not in ("deploy", "registerNode"):
            ops[op].append((gas, lat))
    row = {"K": K, "round_wall_s": wall,
           "round_gas_total": int(sum(g for v in ops.values() for g, _ in v)),
           "n_tx": int(sum(len(v) for v in ops.values()))}
    for op, v in ops.items():
        row[f"{op}_gas"] = float(np.mean([g for g, _ in v]))
        row[f"{op}_ms"] = float(1000 * np.mean([l for _, l in v]))
    row["deploy_gas"] = [g for op, g, _, ok in ch.log if op == "deploy"][0]
    row["register_gas"] = float(np.mean([g for op, g, _, ok in ch.log if op == "registerNode" and ok]))
    out["scaling"].append(row)
    print(row)

# tamper detection: perturb a real serialized XGBoost model in different ways
X = rng.normal(size=(5000, 20)).astype(np.float32); y = (X[:, 0] + rng.normal(size=5000) > 1).astype(int)
m = xgb.XGBClassifier(n_estimators=50, max_depth=4).fit(X, y)
raw = bytes(m.get_booster().save_raw(raw_format="ubj"))
ch = AuditChain(4)
for i, n in enumerate(ch.nodes):
    ch.register(n, f"h{i}")
ch.open_round(1, 0.0)
ch.submit(ch.nodes[0], 1, raw, b"meta", 5000)
for v in ch.nodes[1:]:
    ch.attest(v, 1, ch.nodes[0], raw, 0.8, b"e")
ch.finalize(1)
trials = {"single_bit_flip": 0, "byte_append": 0, "retrained_model": 0}
N = 200
for _ in range(N):
    b = bytearray(raw); i = int(rng.integers(len(b))); b[i] ^= 1 << int(rng.integers(8))
    trials["single_bit_flip"] += not ch.verify(1, ch.nodes[0], bytes(b))
    trials["byte_append"] += not ch.verify(1, ch.nodes[0], raw + rng.bytes(1))
for s in range(20):
    m2 = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=s, subsample=0.9).fit(X, y)
    trials["retrained_model"] += not ch.verify(1, ch.nodes[0], bytes(m2.get_booster().save_raw(raw_format="ubj")))
t = time.perf_counter()
for _ in range(100):
    ch.verify(1, ch.nodes[0], raw)
out["tamper"] = {"trials_bitflip": N, "trials_append": N, "trials_retrained": 20,
                 "detected": trials, "honest_verified": bool(ch.verify(1, ch.nodes[0], raw)),
                 "verify_call_ms": (time.perf_counter() - t) * 10,
                 "model_bytes": len(raw)}
t = time.perf_counter()
for _ in range(100):
    sha256(raw * 100)
out["tamper"]["sha256_ms_per_MB"] = (time.perf_counter() - t) * 10 / (len(raw) * 100 / 1e6)
print(out["tamper"])
(RESULTS / "chain_benchmark.json").write_text(json.dumps(out, indent=1))
