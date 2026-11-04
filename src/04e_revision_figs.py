"""Figures and tables for the revision analyses (reads run JSONs and 04d tables only; no patient data).

Outputs
  results/tables/rev_regimes.csv        data-scarce regimes, full data and care-era partition (mean ± SD over seeds)
  results/tables/rev_attacks.csv        attack scenarios A1–A7 (BFE, oracle, FedAvg family)
  results/tables/rev_admission.csv      admission of the attacker's members by scenario
  figures/fig9_regimes.{png,pdf}        AUROC / AUPRC against training-data fraction
  figures/fig10_attacks.{png,pdf}       stealthier attacks, targeted harm and recalibration poisoning
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGURES, RESULTS

RUNS, TAB = RESULTS / "runs", RESULTS / "tables"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600,
                     "savefig.bbox": "tight", "legend.frameon": False})
NAME = {"Local-XGB": "Local-only XGBoost", "Central-XGB": "Centralised XGBoost*", "FedAvg-MLP": "FedAvg",
        "FedProx-MLP": "FedProx", "BFE-PW": "BFE-PW", "BFE-Stack": "BFE-Stack", "Central-MLP": "Centralised MLP*",
        "Ens-Uniform": "Uniform ensemble", "FedLR": "Federated LR"}
# colour + marker + line style, so series stay distinct in greyscale
STY = {"Local-XGB": ("#7f7f7f", "v", ":"), "Central-XGB": ("#000000", "s", "--"), "FedAvg-MLP": ("#2a78d6", "o", "-."),
       "FedProx-MLP": ("#6aa6e8", "D", "-."), "BFE-PW": ("#c4432f", "^", "-"), "BFE-Stack": ("#e8963a", "P", "-")}
REGIMES = [("small0.05_", "5%"), ("small0.1_", "10%"), ("small0.2_", "20%"), ("", "100%")]


def load(prefix):
    out = []
    for s in range(5):
        f = RUNS / f"{prefix}p0.85_s{s}.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


# ------------------------------------------------------------------ regimes table + figure
rows, curve = [], defaultdict(dict)
for prefix, lab in REGIMES + [("year_", "Care-era sites")]:
    R = load(prefix)
    if not R:
        continue
    ntr = np.mean([sum(r["sites"][s]["train"]["n"] for s in r["sites"]) / 4 for r in R])
    for m in STY:
        a = [r["methods"][m]["auroc"] for r in R]; p = [r["methods"][m]["auprc"] for r in R]
        rows.append({"Setting": lab, "Seeds": len(R), "Mean training admissions per site": f"{ntr:,.0f}",
                     "Method": NAME[m], "AUROC": ms(a), "AUPRC": ms(p)})
        if prefix != "year_":
            curve[m][lab] = (np.mean(a), np.std(a, ddof=1), np.mean(p), np.std(p, ddof=1), ntr)
    adm = [sum(mm["accepted"] for mm in r["chain"]["BFE"]["members"]) for r in R]
    rows.append({"Setting": lab, "Seeds": len(R), "Mean training admissions per site": f"{ntr:,.0f}",
                 "Method": "BFE members admitted (of 8)", "AUROC": f"{min(adm)}–{max(adm)}" if min(adm) != max(adm) else str(adm[0]), "AUPRC": ""})
reg = pd.DataFrame(rows)
reg.to_csv(TAB / "rev_regimes.csv", index=False)
print(reg.to_string())

if curve:
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.0))
    labs = [l for _, l in REGIMES if l in curve["BFE-PW"]]
    xs = np.array([curve["BFE-PW"][l][4] for l in labs])
    for m, (c, mk, ls) in STY.items():
        for k, ax in ((0, axes[0]), (2, axes[1])):
            y = np.array([curve[m][l][k] for l in labs]); e = np.array([curve[m][l][k + 1] for l in labs])
            ax.errorbar(xs, y, yerr=e, color=c, marker=mk, ls=ls, lw=1.4, ms=5, capsize=2, label=NAME[m])
    for ax, yl in zip(axes, ("AUROC (pooled test)", "AUPRC (pooled test)")):
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n({x / 1000:.1f}k)" for l, x in zip(labs, xs)])
        ax.set_xlabel("Training data kept per hospital (mean admissions)"); ax.set_ylabel(yl)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
    axes[0].set_title("(a)", loc="left"); axes[1].set_title("(b)", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig9_regimes.{ext}")
    plt.close(fig)

# ------------------------------------------------------------------ attacks table + figure
main, ext3 = load(""), load("ext3_")
att = []


def add(scen, label, R, key, note=""):
    v = [r["methods"][key] for r in R if key in r["methods"]]
    if not v:
        return
    att.append({"Scenario": scen, "Method": label, "Seeds": len(v), "AUROC": ms([x["auroc"] for x in v]),
                "AUPRC": ms([x["auprc"] for x in v]), "Sensitivity": ms([x["sensitivity"] for x in v]), "Note": note})


def admitted_d(R, label):
    out = []
    for r in R:
        ch = r["chain"].get(label)
        if ch:
            out.append(sum(m["accepted"] for m in ch["members"] if m["owner"] == "D"))
    return out


adm_rows = []
for scen, lab in (("Clean", "BFE"), ("A1 label flip", "A1-BFE"), ("A2 flip + false attestation", "A2-BFE"),
                  ("A4 collusion (C and D)", "A4-BFE")):
    a = admitted_d(main, lab)
    if a:
        adm_rows.append({"Scenario": scen, "Attacker members admitted (of 2), per seed": " ".join(map(str, a))})
add("Clean", "BFE-PW", main, "BFE-PW"); add("Clean", "FedAvg", main, "FedAvg-MLP")
add("A1 label flip", "BFE-PW", main, "A1-BFE-PW"); add("A1 label flip", "FedAvg", main, "A1-FedAvg-MLP")
add("A1 label flip", "Uniform ensemble (no gate)", main, "A1-Ens-Uniform")
add("A2 flip + false attestation", "BFE-PW", main, "A2-BFE-PW")
add("A3 artefact substitution", "BFE-PW (digest check)", main, "A3-BFE-PW")
add("A3 artefact substitution", "Same, no digest check", main, "A3-PW-noverify")
add("A4 collusion (C and D)", "Uniform ensemble (no gate)", main, "A4-Ens-Uniform", "BFE released no model")
if ext3:
    add("Clean", "Federated LR", ext3, "FedLR")
    for nm, lab in (("FedTrim-MLP", "Trimmed-mean FedAvg"), ("FedKrum-MLP", "Multi-Krum FedAvg")):
        add("Clean", lab, ext3, nm); add("A1 label flip", lab, ext3, f"A1-{nm}")
    add("Oracle (D removed by hand)", "BFE-PW", ext3, "Oracle-noD-PW")
    for q in ("A5q20", "A5q40"):
        sc = f"A5 partial flip {q[3:]}%"
        add(sc, "BFE-PW", ext3, f"{q}-BFE-PW"); add(sc, "FedAvg", ext3, f"{q}-FedAvg-MLP")
        a = admitted_d(ext3, f"{q}-BFE")
        adm_rows.append({"Scenario": sc, "Attacker members admitted (of 2), per seed": " ".join(map(str, a))})
    add("A6 targeted flip (female deaths)", "BFE-PW", ext3, "A6-BFE-PW"); add("A6 targeted flip (female deaths)", "FedAvg", ext3, "A6-FedAvg-MLP")
    adm_rows.append({"Scenario": "A6 targeted flip (female deaths)",
                     "Attacker members admitted (of 2), per seed": " ".join(map(str, admitted_d(ext3, "A6-BFE")))})
    add("A7 recalibration poisoning", "BFE-PW (federated Platt)", ext3, "A7-BFE-PW")
    add("A7 recalibration poisoning", "BFE-PW (median-of-sites Platt)", ext3, "A7-BFE-PW-Robust")
    add("Clean", "BFE-PW (median-of-sites Platt)", ext3, "BFE-PW-Robust")
    add("Clean", "BFE-PW (site-specific intercept)", ext3, "BFE-PW-SiteInt")
    # targeted harm: sensitivity and AUROC among female vs male patients
    sex = []
    for key, lab in (("BFE-PW", "Clean BFE-PW"), ("A6-BFE-PW", "A6 BFE-PW"), ("Oracle-noD-PW", "Oracle BFE-PW"),
                     ("FedAvg-MLP", "Clean FedAvg"), ("A6-FedAvg-MLP", "A6 FedAvg")):
        v = [r["subgroup_sex"][key] for r in ext3 if key in r["subgroup_sex"]]
        if v:
            for g in ("F", "M"):
                sex.append({"Model": lab, "Sex": {"F": "Female", "M": "Male"}[g],
                            "AUROC": ms([x[g]["auroc"] for x in v]), "Sensitivity": ms([x[g]["sensitivity"] for x in v]),
                            "PPV": ms([x[g]["ppv"] for x in v])})
    pd.DataFrame(sex).to_csv(TAB / "rev_attack_sex.csv", index=False)
pd.DataFrame(att).to_csv(TAB / "rev_attacks.csv", index=False)
pd.DataFrame(adm_rows).to_csv(TAB / "rev_admission.csv", index=False)
print(pd.DataFrame(att).to_string()); print(pd.DataFrame(adm_rows).to_string())

if ext3:
    cal = TAB / "rev_calibration_summary.csv"
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.9))
    # (a) AUPRC under each attack: BFE-PW vs FedAvg (oracle as reference line)
    sc = [("Clean", "BFE-PW", "FedAvg-MLP", main), ("A1", "A1-BFE-PW", "A1-FedAvg-MLP", main),
          ("A5\n20%", "A5q20-BFE-PW", "A5q20-FedAvg-MLP", ext3), ("A5\n40%", "A5q40-BFE-PW", "A5q40-FedAvg-MLP", ext3),
          ("A6", "A6-BFE-PW", "A6-FedAvg-MLP", ext3)]
    x = np.arange(len(sc)); w = 0.38
    for j, (lab, col, hat) in enumerate((("BFE-PW", "#c4432f", ""), ("FedAvg", "#2a78d6", "///"))):
        vals = [np.mean([r["methods"][s[1 + j]]["auprc"] for r in s[3]]) for s in sc]
        sds = [np.std([r["methods"][s[1 + j]]["auprc"] for r in s[3]], ddof=1) for s in sc]
        axes[0].bar(x + (j - 0.5) * w, vals, w, yerr=sds, color=col, hatch=hat, edgecolor="white", label=lab,
                    error_kw={"lw": 0.8, "capsize": 2})
    orc = np.mean([r["methods"]["Oracle-noD-PW"]["auprc"] for r in ext3])
    axes[0].axhline(orc, color="#000000", ls="--", lw=0.9, label="Oracle (D removed)")
    axes[0].set_xticks(x); axes[0].set_xticklabels([s[0] for s in sc]); axes[0].set_ylabel("AUPRC (pooled test)")
    axes[0].set_ylim(0, 0.44); axes[0].legend(loc="upper center", ncol=2, fontsize=7, columnspacing=0.8, handlelength=1.6)
    # (b) targeted attack: sensitivity among female and male patients
    keys = [("BFE-PW", "Clean"), ("A6-BFE-PW", "A6 BFE"), ("A6-FedAvg-MLP", "A6 FedAvg")]
    for j, (g, col, hat) in enumerate((("F", "#8c564b", ""), ("M", "#bbbbbb", "..."))):
        vals = [np.mean([r["subgroup_sex"][k][g]["sensitivity"] for r in ext3]) for k, _ in keys]
        axes[1].bar(np.arange(3) + (j - 0.5) * w, vals, w, color=col, hatch=hat, edgecolor="white",
                    label={"F": "Female", "M": "Male"}[g])
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels([k[1] for k in keys]); axes[1].set_ylabel("Sensitivity")
    axes[1].set_ylim(0, 0.6); axes[1].legend(loc="upper left", ncol=2, fontsize=7)
    # (c) recalibration poisoning: pooled O/E, standard vs median-of-sites Platt
    if cal.exists():
        c = pd.read_csv(cal); c = c[(c.set == "ext3") & (c.site == "Pooled")].set_index("method")
        items = [("BFE-PW", "Clean"), ("A7-BFE-PW", "A7,\nfederated"), ("A7-BFE-PW-Robust", "A7,\nmedian")]
        vals = [c.loc[k, "o_e_mean"] if k in c.index else np.nan for k, _ in items]
        axes[2].bar(range(3), vals, 0.6, color=["#c4432f", "#555555", "#e8963a"], hatch=["", "xx", ""], edgecolor="white")
        axes[2].axhline(1, color="#000000", ls="--", lw=0.9)
        axes[2].set_xticks(range(3)); axes[2].set_xticklabels([k[1] for k in items]); axes[2].set_ylabel("Observed / expected deaths")
    for ax, t in zip(axes, "abc"):
        ax.set_title(f"({t})", loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig10_attacks.{ext}")
    plt.close(fig)
print("done")
