"""Aggregate run results into manuscript tables and figures."""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from common import FIGURES, RESULTS, SITES, SITE_NAMES, bootstrap_ci, delong_test

RUNS = RESULTS / "runs"
TAB = RESULTS / "tables"; TAB.mkdir(exist_ok=True)

# fixed method -> colour (reference palette, light mode; order never cycled)
COL = {"BFE-PW": "#2a78d6", "FedAvg-MLP": "#eb6834", "Central-XGB": "#1baf7a", "Local-XGB": "#eda100",
       "FedProx-MLP": "#e87ba4", "BFE-Stack": "#008300", "Ens-Uniform": "#6250d6", "Central-MLP": "#e34948"}
MARK = {"BFE-PW": "o", "FedAvg-MLP": "s", "Central-XGB": "^", "Local-XGB": "D", "FedProx-MLP": "v",
        "BFE-Stack": "P", "Ens-Uniform": "X", "Central-MLP": "*"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 600, "savefig.bbox": "tight", "legend.frameon": False})

MAIN = ["Central-XGB", "Central-RF", "Central-MLP", "Local-XGB", "FedAvg-MLP", "FedProx-MLP",
        "Ens-Uniform", "BFE-MLPmembers-PW", "BFE-Stack", "BFE-PW"]
LABEL = {"Central-XGB": "Centralised XGBoost*", "Central-RF": "Centralised random forest*",
         "Central-MLP": "Centralised MLP*", "Local-XGB": "Local-only XGBoost",
         "FedAvg-MLP": "FedAvg", "FedProx-MLP": "FedProx", "Ens-Uniform": "Uniform ensemble",
         "BFE-MLPmembers-PW": "BFE with MLP members (ablation)", "BFE-Stack": "BFE-Stack",
         "BFE-PW": "BFE-PW"}
# line styles so curves stay distinct in greyscale
LS = {"BFE-PW": "-", "BFE-Stack": (0, (5, 1.5)), "FedAvg-MLP": "-.", "FedProx-MLP": (0, (3, 1, 1, 1, 1, 1)),
      "Central-XGB": "--", "Local-XGB": ":", "Ens-Uniform": (0, (1, 1)), "Central-MLP": (0, (8, 2))}
METRICS = ["auroc", "auprc", "f1", "accuracy", "sensitivity", "specificity", "brier", "ece"]


def load_runs():
    runs = []
    for f in sorted(RUNS.glob("p*_s*.json")):
        r = json.loads(f.read_text())
        r["_file"] = f.stem
        runs.append(r)
    return runs


def ms(vals, k=3):
    vals = np.asarray(vals, float)
    return f"{vals.mean():.{k}f} ± {vals.std(ddof=1):.{k}f}" if len(vals) > 1 else f"{vals.mean():.{k}f}"


runs = load_runs()
by_p = defaultdict(list)
for r in runs:
    by_p[round(r["purity"], 2)].append(r)
print({p: len(v) for p, v in by_p.items()})
PRIMARY = 0.85
prim = by_p[PRIMARY]

# --------------------------------------------------------------- Table 1: site characteristics (seed 0)
from common import load_cohort, prepare, build_features
d = load_cohort()
X, y, site, split = prepare(PRIMARY, 0, d, build_features(d))
lab_mean_cols = [c for c in X.columns if c.endswith("_mean")]
rows = []
for s in SITES + ["All"]:
    m = np.ones(len(d), bool) if s == "All" else site == s
    dd = d[m]
    rows.append({
        "Hospital": "Pooled" if s == "All" else SITE_NAMES[s],
        "Admissions": f"{m.sum():,}", "Patients": f"{dd.subject_id.nunique():,}",
        "Deaths, n (%)": f"{int(dd.label.sum()):,} ({100 * dd.label.mean():.2f})",
        "Age, median [IQR]": f"{dd.age.median():.0f} [{dd.age.quantile(.25):.0f}–{dd.age.quantile(.75):.0f}]",
        "Female, %": f"{100 * (dd.gender == 'F').mean():.1f}",
        "Emergency admission, %": f"{100 * dd.admission_type.str.contains('EMER').mean():.1f}",
        "Creatinine max ≥2.0 mg/dL, %": f"{100 * (dd.creatinine_max >= 2).mean():.1f}",
        "Prior CKD, %": f"{100 * dd.hx_ckd.mean():.1f}",
        "Lab features missing, %": f"{100 * X.loc[m, lab_mean_cols].isna().values.mean():.1f}",
    })
t1 = pd.DataFrame(rows)
t1.to_csv(TAB / "table1_sites.csv", index=False)

# IID reference characteristics
Xi, yi, site_i, _ = prepare(0.0, 0, d, build_features(d))
het = []
for p in (0.0, 0.5, 0.85, 1.0):
    _, yy, ss, _ = prepare(p, 0, d, build_features(d))
    prev = [yy[ss == s].mean() for s in SITES]
    age = [d.age[ss == s].mean() for s in SITES]
    sizes = [(ss == s).mean() for s in SITES]
    het.append({"purity": p, "mortality_range_pct": f"{100*min(prev):.2f}–{100*max(prev):.2f}",
                "mean_age_range": f"{min(age):.1f}–{max(age):.1f}",
                "site_share_range_pct": f"{100*min(sizes):.1f}–{100*max(sizes):.1f}"})
pd.DataFrame(het).to_csv(TAB / "heterogeneity_by_purity.csv", index=False)
del X, Xi

# --------------------------------------------------------------- Table 2: main results (mean ± SD over seeds)
def summary_table(runs_, methods, path):
    out = []
    for mth in methods:
        vals = {k: [r["methods"][mth][k] for r in runs_ if mth in r["methods"]] for k in METRICS}
        if not vals["auroc"]:
            continue
        row = {"Method": LABEL.get(mth, mth)}
        for k in METRICS:
            row[k.upper() if k in ("auroc", "auprc", "f1", "ece") else k.capitalize()] = ms(vals[k], 4 if k in ("brier", "ece") else 3)
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(path, index=False)
    return df


t2 = summary_table(prim, MAIN, TAB / "table2_main_noniid.csv")
t2i = summary_table(by_p[0.0], MAIN, TAB / "table2b_iid.csv")
print(t2.to_string()); print(t2i.to_string())

# DeLong per seed + bootstrap CI (seed 0)
stats = []
for r in prim:
    P_ = np.load(RUNS / f"{r['_file']}_preds.npz")
    yt = P_["y"]
    for comp in ("FedAvg-MLP", "FedProx-MLP", "Local-XGB", "Ens-Uniform", "Central-XGB", "BFE-Stack"):
        a1, a2, z, pv = delong_test(yt, P_["BFE-PW"], P_[comp])
        stats.append({"seed": r["seed"], "comparison": f"BFE-PW vs {comp}", "auroc_bfe": a1, "auroc_comp": a2,
                      "diff": a1 - a2, "z": z, "p": pv})
st = pd.DataFrame(stats)
st.to_csv(TAB / "delong_noniid.csv", index=False)
print(st.groupby("comparison")[["diff", "p"]].agg(["mean", "max"]))

r0 = [r for r in prim if r["seed"] == 0][0]
P0 = np.load(RUNS / f"{r0['_file']}_preds.npz")
ci = []
for mth in MAIN:
    if mth not in P0.files:
        continue
    lo, hi = bootstrap_ci(P0["y"], P0[mth], roc_auc_score, 1000, 0)
    lo2, hi2 = bootstrap_ci(P0["y"], P0[mth], average_precision_score, 1000, 0)
    ci.append({"Method": LABEL[mth], "AUROC": f"{roc_auc_score(P0['y'], P0[mth]):.3f} ({lo:.3f}–{hi:.3f})",
               "AUPRC": f"{average_precision_score(P0['y'], P0[mth]):.3f} ({lo2:.3f}–{hi2:.3f})"})
pd.DataFrame(ci).to_csv(TAB / "bootstrap_ci_seed0.csv", index=False)

# --------------------------------------------------------------- Table 3: per-site AUROC / AUPRC (non-IID)
rows = []
for mth in ["Local-XGB", "FedAvg-MLP", "FedProx-MLP", "Ens-Uniform", "BFE-PW", "Central-XGB"]:
    row = {"Method": LABEL[mth]}
    for s in SITES:
        row[f"{s} AUROC"] = ms([r["per_site"][mth][s]["auroc"] for r in prim])
        row[f"{s} AUPRC"] = ms([r["per_site"][mth][s]["auprc"] for r in prim])
    rows.append(row)
pd.DataFrame(rows).to_csv(TAB / "table3_per_site.csv", index=False)

# --------------------------------------------------------------- subgroup audit (sex, age band, race group; non-IID)
SUB_METHODS = ["BFE-PW", "BFE-Stack", "FedAvg-MLP", "FedProx-MLP", "Local-XGB", "Central-XGB"]
age_band = pd.cut(d.age, [0, 50, 70, 85, 200], right=False, labels=["18–49", "50–69", "70–84", "≥85"]).astype(str)
groups = {"Sex": d.gender.map({"F": "Female", "M": "Male"}).values, "Age": age_band.values,
          "Race": d.race_group.str.title().values}
sub = []
for r in prim:
    _, yy, ss, sp = prepare(PRIMARY, r["seed"], d, build_features(d))
    tidx = np.concatenate([np.where((ss == s) & (sp == "test"))[0] for s in SITES])
    P_ = np.load(RUNS / f"{r['_file']}_preds.npz")
    assert np.array_equal(P_["y"], yy[tidx])
    for gname, gv in groups.items():
        gt = gv[tidx]
        for g in np.unique(gt):
            m = gt == g
            if P_["y"][m].sum() < 20:
                continue
            for mth in SUB_METHODS:
                p = P_[mth][m]
                sub.append({"seed": r["seed"], "attribute": gname, "group": g, "method": mth, "n": int(m.sum()),
                            "events": int(P_["y"][m].sum()), "auroc": roc_auc_score(P_["y"][m], p),
                            "auprc": average_precision_score(P_["y"][m], p),
                            "o_e": P_["y"][m].mean() / p.mean()})
sub = pd.DataFrame(sub)
sub.to_csv(TAB / "subgroups_long.csv", index=False)
sg = sub.groupby(["attribute", "group", "method"]).agg(n=("n", "mean"), events=("events", "mean"),
                                                      auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                                                      auprc=("auprc", "mean"), o_e=("o_e", "mean")).reset_index()
sg.to_csv(TAB / "table6_subgroups.csv", index=False)
gap = sg.groupby(["attribute", "method"]).auroc.agg(lambda v: v.max() - v.min()).unstack()
gap.to_csv(TAB / "subgroup_auroc_gap.csv")
print(gap.round(3).to_string())

order = [("Sex", g) for g in ("Female", "Male")] + [("Age", g) for g in ("18–49", "50–69", "70–84", "≥85")] + \
        [("Race", g) for g in sorted(sg[sg.attribute == "Race"].group.unique())]
order = [o for o in order if ((sg.attribute == o[0]) & (sg.group == o[1])).any()]
fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True, constrained_layout=True)
show_sg = ["BFE-PW", "BFE-Stack", "FedAvg-MLP", "Central-XGB"]
w = 0.2
oe_sd = sub.groupby(["attribute", "group", "method"]).o_e.std().reset_index()
for k, mth in enumerate(show_sg):
    vals = [sg[(sg.attribute == a) & (sg.group == g) & (sg.method == mth)] for a, g in order]
    sds = [oe_sd[(oe_sd.attribute == a) & (oe_sd.group == g) & (oe_sd.method == mth)].o_e.iloc[0] for a, g in order]
    xs_ = np.arange(len(order)) + (k - 1.5) * w
    axes[0].errorbar(xs_, [v.auroc.iloc[0] for v in vals], yerr=[v.auroc_sd.iloc[0] for v in vals], fmt=MARK[mth],
                     color=COL[mth], ms=4.5, capsize=1.5, lw=1, label=LABEL[mth])
    axes[1].errorbar(xs_, [v.o_e.iloc[0] for v in vals], yerr=sds, fmt=MARK[mth], color=COL[mth], ms=4.5,
                     capsize=1.5, lw=1)
axes[1].axhline(1, color=INK2, lw=0.9, ls="--")
for ax in axes:
    for x in (1.5, 5.5):
        ax.axvline(x, color=INK2, lw=0.8, ls=":")
for xc, t in ((0.5, "Sex"), (3.5, "Age band (years)"), (5.5 + (len(order) - 6) / 2, "Race group")):
    axes[0].text(xc, 0.03, t, transform=axes[0].get_xaxis_transform(), ha="center", fontsize=8, color=INK2,
                 weight="bold")
axes[1].set_xticks(range(len(order)), [g for _, g in order], rotation=30, ha="right")
axes[0].set_ylabel("Pooled-test AUROC"); axes[1].set_ylabel("Observed / expected deaths")
axes[0].legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
axes[0].set_ylim(sg.auroc.min() - 0.035, None)
for ax, t in zip(axes, "ab"):
    ax.set_title(f"({t})", loc="left", fontsize=10, color=INK)
fig.savefig(FIGURES / "fig8_subgroups.png"); fig.savefig(FIGURES / "fig8_subgroups.pdf"); plt.close(fig)

# --------------------------------------------------------------- Table 4: adversarial scenarios
att_rows = []
scen = [("No attack", "BFE-PW", "Ens-Uniform", "FedAvg-MLP"),
        ("A1 label flipping", "A1-BFE-PW", "A1-Ens-Uniform", "A1-FedAvg-MLP"),
        ("A2 flipping + adversarial attestation", "A2-BFE-PW", None, None),
        ("A3 artefact substitution", "A3-BFE-PW", "A3-Ens-Uniform-noverify", None),
        ("A4 two colluding hospitals", "A4-BFE-PW", "A4-Ens-Uniform", None)]
for name, bfe, uni, fed in scen:
    row = {"Scenario": name}
    for lab, key in (("BFE-PW (ledger)", bfe), ("Uniform ensemble", uni), ("FedAvg", fed)):
        if key is None:
            row[f"{lab} AUROC"] = row[f"{lab} AUPRC"] = "–"
            continue
        v = [r["methods"][key] for r in prim if key in r["methods"]]
        if not v:
            row[f"{lab} AUROC"] = row[f"{lab} AUPRC"] = "no global model"
            continue
        row[f"{lab} AUROC"] = ms([x["auroc"] for x in v]); row[f"{lab} AUPRC"] = ms([x["auprc"] for x in v])
    if name.startswith("A3"):
        v = [r["methods"]["A3-PW-noverify"] for r in prim if "A3-PW-noverify" in r["methods"]]
        row["PW without hash verification AUROC"] = ms([x["auroc"] for x in v])
        row["PW without hash verification AUPRC"] = ms([x["auprc"] for x in v])
    att_rows.append(row)
pd.DataFrame(att_rows).to_csv(TAB / "table4_attacks.csv", index=False)

# admission decisions under attack
dec = []
for lab in ("BFE", "A1-BFE", "A2-BFE", "A4-BFE"):
    for r in prim:
        if lab in r["chain"]:
            for m in r["chain"][lab]["members"]:
                dec.append({"scenario": lab, "seed": r["seed"], "owner": m["owner"], "family": m["family"],
                            "accepted": m["accepted"], "median_peer_auroc": m["median_peer_auroc"]})
dec = pd.DataFrame(dec)
dsum = dec.groupby(["scenario", "owner", "family"]).agg(accept_rate=("accepted", "mean"),
                                                     median_auroc=("median_peer_auroc", "mean")).reset_index()
dsum.to_csv(TAB / "admission_decisions.csv", index=False)
a3 = [r["attack"] for r in prim if "attack" in r]
pd.DataFrame(a3).to_csv(TAB / "a3_detection.csv", index=False)

# --------------------------------------------------------------- severity table/figure
sev_methods = ["Central-XGB", "BFE-Stack", "BFE-PW", "Local-XGB", "FedAvg-MLP", "FedProx-MLP", "Ens-Uniform"]
sev = []
for p in sorted(by_p):
    for mth in sev_methods:
        a = [r["methods"][mth]["auroc"] for r in by_p[p]]; b = [r["methods"][mth]["auprc"] for r in by_p[p]]
        sev.append({"purity": p, "method": mth, "auroc_mean": np.mean(a), "auroc_sd": np.std(a, ddof=1) if len(a) > 1 else 0,
                    "auprc_mean": np.mean(b), "auprc_sd": np.std(b, ddof=1) if len(b) > 1 else 0, "n_seeds": len(a)})
sev = pd.DataFrame(sev); sev.to_csv(TAB / "severity.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.0), constrained_layout=True)
for ax, met, lab in ((axes[0], "auroc", "AUROC"), (axes[1], "auprc", "AUPRC")):
    for j, mth in enumerate(sev_methods):
        s_ = sev[sev.method == mth]
        ax.errorbar(s_.purity + (j - 3) * 0.008, s_[f"{met}_mean"], yerr=s_[f"{met}_sd"], color=COL[mth], marker=MARK[mth],
                    ms=5, lw=1.6, capsize=2, label=LABEL[mth])
    ax.set_xlabel("Skew purity p (0 = IID, 1 = fully skewed)"); ax.set_ylabel(f"Pooled-test {lab}")
    ax.set_xticks(sorted(by_p))
axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
for ax, t in zip(axes, "ab"):
    ax.set_title(f"({t})", loc="left", fontsize=10, color=INK)
fig.savefig(FIGURES / "fig4_severity.png"); fig.savefig(FIGURES / "fig4_severity.pdf"); plt.close(fig)

# --------------------------------------------------------------- Fig: ROC / PR / calibration (seed 0, non-IID)
show = ["Central-XGB", "Local-XGB", "FedAvg-MLP", "FedProx-MLP", "Ens-Uniform", "BFE-Stack", "BFE-PW"]
cal_max = 0
fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.9))
yt = P0["y"]
for mth in show:
    p = P0[mth]
    fpr, tpr, _ = roc_curve(yt, p); pr, rc, _ = precision_recall_curve(yt, p)
    lw_, z_ = (2.4, 5) if mth == "BFE-PW" else (1.3, 2)
    axes[0].plot(fpr, tpr, color=COL[mth], ls=LS[mth], lw=lw_, zorder=z_, label=LABEL[mth])
    axes[1].plot(rc, pr, color=COL[mth], ls=LS[mth], lw=lw_, zorder=z_)
    fy, fx = calibration_curve(yt, p, n_bins=10, strategy="quantile")
    axes[2].plot(fx, fy, color=COL[mth], ls=LS[mth], marker=MARK[mth], ms=4, lw=lw_ - 0.2, zorder=z_)
    cal_max = max(cal_max, fx.max(), fy.max())
axes[0].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--"); axes[0].set_xlabel("1 − specificity"); axes[0].set_ylabel("Sensitivity")
axes[1].axhline(yt.mean(), color=INK2, lw=0.8, ls="--"); axes[1].set_xlabel("Recall (sensitivity)"); axes[1].set_ylabel("Precision (PPV)")
lim = max(P0["BFE-PW"].max(), 0.6)
axes[2].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--"); cl = round(cal_max * 1.1, 2); axes[2].set_xlim(0, cl); axes[2].set_ylim(0, cl)
axes[2].set_xlabel("Predicted risk"); axes[2].set_ylabel("Observed mortality")
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
for ax, t in zip(axes, "abc"):
    ax.set_title(f"({t})", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(FIGURES / "fig3_roc_pr_calibration.png"); fig.savefig(FIGURES / "fig3_roc_pr_calibration.pdf"); plt.close(fig)

# --------------------------------------------------------------- Fig: decision curve (seed 0)
fig, ax = plt.subplots(figsize=(4.5, 3.4))
th = np.linspace(0.01, 0.30, 59)
prev = yt.mean()
for mth in ["Central-XGB", "Local-XGB", "FedAvg-MLP", "BFE-Stack", "BFE-PW"]:
    p = P0[mth]
    nb = [((p >= t) & (yt == 1)).mean() - ((p >= t) & (yt == 0)).mean() * t / (1 - t) for t in th]
    ax.plot(th, nb, color=COL[mth], ls=LS[mth], lw=1.6, label=LABEL[mth])
ax.plot(th, [prev - (1 - prev) * t / (1 - t) for t in th], color=INK2, lw=0.9, ls="--", label="Treat all")
ax.axhline(0, color=INK2, lw=0.9, ls=":", label="Treat none")
ax.set_ylim(-0.002, prev * 1.05); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.legend(fontsize=8)
fig.savefig(FIGURES / "fig6_decision_curve.png"); fig.savefig(FIGURES / "fig6_decision_curve.pdf"); plt.close(fig)

# --------------------------------------------------------------- Fig: heterogeneity + cross-site matrix
fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.6)); axes = axes.ravel()
_, yy, ss, _ = prepare(PRIMARY, 0, d, build_features(d))
site_col = "#2a78d6"
ages = [d.age[ss == s].values for s in SITES]
bp = axes[0].boxplot(ages, tick_labels=SITES, showfliers=False, patch_artist=True, widths=0.5,
                     medianprops={"color": INK})
for b in bp["boxes"]:
    b.set_facecolor("#c9dcf3"); b.set_edgecolor(site_col)
axes[0].set_ylabel("Age at admission (years)"); axes[0].set_xlabel("Hospital")
mort = [100 * yy[ss == s].mean() for s in SITES]
miss = [float(t1.loc[i, "Lab features missing, %"]) for i in range(4)]
xs = np.arange(4)
for ax, vals, lab, fmt in ((axes[1], mort, "In-hospital mortality (%)", "{:.2f}"),
                           (axes[2], miss, "Laboratory features missing (%)", "{:.0f}")):
    ax.bar(xs, vals, 0.6, color=site_col)
    for x, v in zip(xs, vals):
        ax.text(x, v * 1.02, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(xs, SITES); ax.set_xlabel("Hospital"); ax.set_ylabel(lab); ax.set_ylim(0, max(vals) * 1.18)
M = np.array([[np.mean([r["cross_site_auroc"][f"{a}->{b}"] for r in prim]) for b in SITES] for a in SITES])
im = axes[3].imshow(M, cmap="Blues", vmin=M.min() - 0.02, vmax=M.max())
for i in range(4):
    for j in range(4):
        axes[3].text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9,
                     color="white" if M[i, j] > (M.min() + M.max()) / 2 else INK)
axes[3].set_xticks(range(4), SITES); axes[3].set_yticks(range(4), SITES); axes[3].grid(False)
axes[3].set_xlabel("Evaluated on hospital (test)"); axes[3].set_ylabel("Local XGBoost trained at")
fig.colorbar(im, ax=axes[3], fraction=0.046, label="AUROC")
for ax, t in zip(axes, "abcd"):
    ax.set_title(f"({t})", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(FIGURES / "fig2_heterogeneity.png"); fig.savefig(FIGURES / "fig2_heterogeneity.pdf"); plt.close(fig)
np.savetxt(TAB / "cross_site_auroc.csv", M, delimiter=",", fmt="%.4f", header=",".join(SITES))

# --------------------------------------------------------------- Fig: attacks (AUPRC and ECE)
rob = {}
for f in sorted(RUNS.glob("robust_p0.85_s*.json")):
    rr = json.loads(f.read_text()); rob[rr["seed"]] = rr["methods"]


def att_vals(key, met):
    if key.endswith("FedMedian-MLP"):
        return [m[key][met] for m in rob.values() if key in m]
    return [r["methods"][key][met] for r in prim if key in r["methods"]]


groups = ["No attack", "A1", "A2", "A3", "A4"]
series = [("BFE-PW (ledger)", ["BFE-PW", "A1-BFE-PW", "A2-BFE-PW", "A3-BFE-PW", "A4-BFE-PW"], "#2a78d6"),
          ("Uniform ensemble", ["Ens-Uniform", "A1-Ens-Uniform", None, "A3-Ens-Uniform-noverify", "A4-Ens-Uniform"], "#6250d6"),
          ("PW, no hash check", [None, None, None, "A3-PW-noverify", None], "#1baf7a"),
          ("FedAvg (MLP)", ["FedAvg-MLP", "A1-FedAvg-MLP", None, None, None], "#eb6834")]
if rob:
    series.append(("FedAvg, coordinate-wise median", ["FedMedian-MLP", "A1-FedMedian-MLP", None, None, None], "#eda100"))
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3), constrained_layout=True)
w = 0.8 / len(series)
for ax, met, lab in ((axes[0], "auprc", "Pooled-test AUPRC"), (axes[1], "ece", "Expected calibration error (log)")):
    for k, (sl, keys, c) in enumerate(series):
        first = next(i for i, kk in enumerate(keys) if kk)
        for g, key in enumerate(keys):
            if key is None:
                continue
            v = att_vals(key, met)
            x = g + (k - (len(series) - 1) / 2) * w
            if not v:
                ax.text(x, 0.5, "no model released", transform=ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=6, color=INK2, rotation=90)
                continue
            ax.bar(x, np.mean(v), w * 0.92, yerr=np.std(v, ddof=1) if len(v) > 1 else 0, color=c,
                   error_kw={"lw": 0.8, "capsize": 2}, label=sl if (g == first and ax is axes[0]) else None)
    ax.set_xticks(range(5), groups); ax.set_ylabel(lab)
axes[1].set_yscale("log")
fig.legend(fontsize=7, ncol=len(series), loc="lower center", bbox_to_anchor=(0.5, 1.0))
for ax, t in zip(axes, "ab"):
    ax.set_title(f"({t})", loc="left", fontsize=10, color=INK)
fig.savefig(FIGURES / "fig5_attacks.png"); fig.savefig(FIGURES / "fig5_attacks.pdf"); plt.close(fig)

# --------------------------------------------------------------- chain tables / figure
cb = json.loads((RESULTS / "chain_benchmark.json").read_text())
sc = pd.DataFrame(cb["scaling"])
sc.to_csv(TAB / "chain_scaling.csv", index=False)
ops = defaultdict(list)
for r in prim:
    for op, v in r["chain"]["ops"].items():
        ops[op].append(v)
op_rows = [{"Operation": op, "Calls per run": int(np.mean([v["count"] for v in vs])),
            "Gas (mean)": int(np.mean([v["gas_mean"] for v in vs])),
            "Latency ms (mean)": round(float(np.mean([v["latency_ms_mean"] for v in vs])), 1)} for op, vs in ops.items()]
pd.DataFrame(op_rows).to_csv(TAB / "table5_chain_ops.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.0))
axes[0].plot(sc.K, sc.round_gas_total / 1e6, color="#2a78d6", marker="o", lw=1.6, label="Total gas per round")
axes[0].plot(sc.K, sc.finalizeRound_gas / 1e6, color="#eb6834", marker="s", lw=1.6, label="finalizeRound")
for ax in axes:
    ax.set_xscale("log", base=2); ax.set_yscale("log")
axes[0].set_xlabel("Number of hospitals K"); axes[0].set_ylabel("Gas per round (millions, log)"); axes[0].set_xticks(sc.K, sc.K)
axes[0].legend(fontsize=8)
axes[1].plot(sc.K, sc.n_tx, color="#2a78d6", marker="o", lw=1.6)
axes[1].set_xlabel("Number of hospitals K"); axes[1].set_ylabel("Transactions per round (log)"); axes[1].set_xticks(sc.K, sc.K)
for ax, t in zip(axes, "ab"):
    ax.set_title(f"({t})", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(FIGURES / "fig7_chain_scaling.png"); fig.savefig(FIGURES / "fig7_chain_scaling.pdf"); plt.close(fig)

# --------------------------------------------------------------- compute / communication cost
cost = []
for r in prim:
    T = r["timing"]
    cost.append({"seed": r["seed"],
                 "local_trees_s": T["local_trees_total_s"], "fedavg_s": T["FedAvg-MLP_s"], "fedprox_s": T["FedProx-MLP_s"],
                 "central_xgb_s": T["central_xgb_s"], "bfe_protocol_s": T["bfe_protocol_total_s"],
                 "bfe_chain_s": r["chain"]["BFE"]["timing"]["protocol_s"],
                 "bfe_peer_eval_s": r["chain"]["BFE"]["timing"]["peer_eval_s"],
                 "xgb_bytes": r["model_bytes"]["xgb_mean"], "rf_bytes": r["model_bytes"]["rf_mean"],
                 "mlp_bytes": r["model_bytes"]["mlp"]})
cost = pd.DataFrame(cost); cost.to_csv(TAB / "cost.csv", index=False)
print(cost.mean().round(2).to_string())
print(dsum.to_string())
print(pd.DataFrame(att_rows).to_string())
