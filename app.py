"""
Supplementary Code S1
Reproduces every number in Tables 9, 11 and 12, the correlations in Section 4, Figure 1 and the
counts in Figure 2 from Supplementary_Data_S1_Extraction_Matrix.csv.

Usage:  python Supplementary_Code_S1_Analysis.py [path/to/Supplementary_Data_S1_Extraction_Matrix.csv] [outdir]

Notes
- Values are those tabulated in manuscript Tables 5-8. Their source-level provenance (R/D/E) is
  still UNVERIFIED and must be checked against the cited articles before any formal meta-analysis.
- Metric families are kept separate: accuracy-type percentages, relative time reduction, relative
  speed increase, energy reduction, other resource reduction, absolute latency. They are never pooled.
- Equivalent fold speed-up converts a relative time reduction r (%) to 1/(1 - r/100) and a relative
  speed increase s (%) to 1 + s/100; this assumes the same workload before and after.
- Bootstrap: 10,000 resamples of the mean, percentile intervals, numpy default_rng(seed=42),
  one fresh generator per call so each interval is reproducible on its own.
"""
import sys, os
import numpy as np, pandas as pd
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else "Supplementary_Data_S1_Extraction_Matrix.csv"
outdir = sys.argv[2] if len(sys.argv) > 2 else "."
df = pd.read_csv(path)
df["duplicate_of"] = df["duplicate_of"].fillna("")
uniq = df[df["duplicate_of"] == ""].copy()

def fold(row):
    if pd.notna(row["time_red"]):
        return 1.0 / (1.0 - row["time_red"] / 100.0)
    if pd.notna(row["speed_inc"]):
        return 1.0 + row["speed_inc"] / 100.0
    return np.nan
uniq["fold_speedup"] = uniq.apply(fold, axis=1)

def boot_ci(x, n_boot=10000, seed=42, stat=np.mean):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = np.array([stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return np.percentile(b, [2.5, 97.5])

def describe(x, gm=False):
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return dict(n=0)
    d = dict(n=len(x), mean=x.mean(), median=x.median(), sd=x.std(ddof=1) if len(x) > 1 else np.nan,
             q1=x.quantile(.25), q3=x.quantile(.75))
    if len(x) >= 3:
        lo, hi = boot_ci(x)
        d.update(ci_lo=lo, ci_hi=hi)
    if gm:
        d["geo_mean"] = float(np.exp(np.log(x).mean()))
        lo, hi = boot_ci(np.log(x), stat=np.mean)
        d.update(gm_lo=float(np.exp(lo)), gm_hi=float(np.exp(hi)))
    return d

FAMS = [("acc", "Accuracy-type level (%)"), ("time_red", "Relative time reduction (%)"),
        ("speed_inc", "Relative speed increase (%)"), ("fold_speedup", "Equivalent fold speed-up (x)"),
        ("energy_red", "Energy reduction (%)"), ("resource_red", "Other resource reduction (%)"),
        ("latency_ms", "Absolute latency (ms)")]

sets = {
    "A_all_unique": uniq,
    "B_primary_eligible": uniq[uniq.eligible_primary == 1],
    "C_quantum_methods_only": uniq[uniq.screen_code == "Q"],
}
def no_bounded(d):
    d = d.copy()
    for f in ["acc", "time_red", "speed_inc", "energy_red", "resource_red", "latency_ms"]:
        d.loc[d[f + "_form"].isin(["<", ">", "\u2264", "\u2265"]), f] = np.nan
    return d
sets["D_primary_no_bounded"] = no_bounded(sets["B_primary_eligible"])

lines = []
def out(s=""):
    print(s); lines.append(s)

out("== Corpus counts (Figure 2)")
out(f"Table entries: {len(df)}; unique studies: {len(uniq)}")
for k, v in uniq.screen_code.value_counts().sort_index().items():
    out(f"  screen {k}: {v}")
out(f"Primary quantitative set: {int(uniq.eligible_primary.sum())}")
prim = sets["B_primary_eligible"]
out("Primary set by domain: " + str(prim.domain.value_counts().to_dict()))
out("Primary set by category: " + str(prim.screen_code.value_counts().to_dict()))

out("\n== Table 9: primary set (eligible, unique)")
for f, lab in FAMS:
    d = describe(prim[f], gm=(f == "fold_speedup"))
    out(f"{lab}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()))

out("\n== Table 11: sensitivity")
for sname, sd in sets.items():
    for f, lab in FAMS:
        d = describe(sd[f])
        if d["n"] == 0:
            continue
        ci = f"{d['ci_lo']:.2f}-{d['ci_hi']:.2f}" if "ci_lo" in d else "NA"
        out(f"{sname} | {lab} | n={d['n']} | mean={d['mean']:.2f} | median={d['median']:.2f} | CI={ci}")

out("\n== Table 12: group-wise comparisons within commensurate families (primary set)")
for f, lab in [("acc", "Accuracy-type level (%)"), ("fold_speedup", "Equivalent fold speed-up (x)")]:
    for g in ["screen_code", "domain"]:
        groups = {k: v[f].dropna().values for k, v in prim.groupby(g)}
        groups = {k: v for k, v in groups.items() if len(v) >= 3}
        for k, v in groups.items():
            d = describe(v)
            out(f"{lab} | {g}={k} | n={d['n']} | mean={d['mean']:.2f} | median={d['median']:.2f} | CI={d['ci_lo']:.2f}-{d['ci_hi']:.2f}")
        if len(groups) >= 2:
            H, p = stats.kruskal(*groups.values())
            out(f"{lab} | {g} | Kruskal-Wallis H={H:.3f}, df={len(groups)-1}, p={p:.3f}")

out("\n== Spearman correlations (primary set, pairwise complete, n>=5)")
CORR = ["acc", "fold_speedup", "energy_red", "latency_ms"]
corr = {}
for i, a in enumerate(CORR):
    for b in CORR[i + 1:]:
        sub = prim[[a, b]].dropna()
        if len(sub) >= 5:
            rho, p = stats.spearmanr(sub[a], sub[b])
            corr[(a, b)] = (rho, p, len(sub))
            out(f"{a} vs {b}: n={len(sub)}, rho={rho:.3f}, p={p:.3f}")
        else:
            corr[(a, b)] = (np.nan, np.nan, len(sub))
            out(f"{a} vs {b}: n={len(sub)} (<5, not estimated)")

out("\n== Spearman sensitivity: accuracy vs equivalent fold speed-up")
for sname, sd in sets.items():
    sub = sd[["acc", "fold_speedup"]].dropna()
    rho, p = stats.spearmanr(sub["acc"], sub["fold_speedup"])
    out(f"{sname}: n={len(sub)}, rho={rho:.3f}, p={p:.3f}")
est = [v for v in corr.values() if not np.isnan(v[0])]
out(f"Estimable correlations in primary set: {len(est)}; Bonferroni threshold = {0.05/len(est):.4f}")

with open(os.path.join(outdir, "S1_results.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

# ---------- Figure 1 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(2, 2, figsize=(10, 8.6))
names = {"acc": "Accuracy", "fold_speedup": "Fold\nspeed-up", "energy_red": "Energy\nreduction", "latency_ms": "Latency"}
M = np.full((4, 4), np.nan); N = np.zeros((4, 4), int)
for i, a in enumerate(CORR):
    M[i, i] = 1; N[i, i] = prim[a].notna().sum()
    for j, b in enumerate(CORR):
        if i < j:
            rho, p, n = corr[(a, b)]
            M[i, j] = M[j, i] = rho; N[i, j] = N[j, i] = n
a0 = ax[0, 0]
im = a0.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(4):
    for j in range(4):
        t = f"{M[i,j]:.2f}\n(n={N[i,j]})" if not np.isnan(M[i, j]) else f"n={N[i,j]}\n(<5)"
        a0.text(j, i, t, ha="center", va="center", fontsize=8, color=("white" if (not np.isnan(M[i, j]) and abs(M[i, j]) > 0.6) else "black"))
a0.set_xticks(range(4)); a0.set_yticks(range(4))
a0.set_xticklabels([names[c] for c in CORR]); a0.set_yticklabels([names[c] for c in CORR])
a0.set_title("(a) Spearman \u03c1, primary set, pairwise complete")
fig.colorbar(im, ax=a0, fraction=0.046)

a1 = ax[0, 1]
order = ["Q", "QI", "QO"]; lab = {"Q": "Quantum\nmethod", "QI": "Quantum-\ninspired", "QO": "Classical AI,\nquantum-oriented"}
rng = np.random.default_rng(1)
for k, c in enumerate(order):
    v = prim.loc[prim.screen_code == c, "acc"].dropna().values
    a1.scatter(k + rng.uniform(-.12, .12, len(v)), v, s=18, alpha=.7, color="#4C72B0")
    if len(v) >= 3:
        lo, hi = boot_ci(v)
        a1.errorbar(k + .3, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="D", color="#C44E52", capsize=4)
    a1.text(k, 86.2, f"n={len(v)}", ha="center", fontsize=8)
a1.set_xticks(range(3)); a1.set_xticklabels([lab[c] for c in order]); a1.set_ylim(85.5, 100)
a1.set_ylabel("Accuracy-type level (%)"); a1.set_title("(b) Accuracy by modality class: mean \u00b1 95% CI")

a2 = ax[1, 0]
data = [prim[f].dropna().values for f in ["time_red", "speed_inc", "energy_red", "resource_red"]]
a2.boxplot(data, showmeans=True)
a2.set_xticks([1, 2, 3, 4]); a2.set_xticklabels([f"Time\nreduction\n(n={len(data[0])})", f"Speed\nincrease\n(n={len(data[1])})",
                       f"Energy\nreduction\n(n={len(data[2])})", f"Other resource\nreduction\n(n={len(data[3])})"])
a2.set_ylabel("Relative change (%)"); a2.set_title("(c) Relative-change families, kept separate")

a3 = ax[1, 1]
cats = ["Q", "QI", "QO", "NQ", "NAI", "NE"]
counts = [int((uniq.screen_code == c).sum()) for c in cats]
cols = ["#4C72B0"] * 3 + ["#BBBBBB"] * 3
bars = a3.bar(cats, counts, color=cols)
for b_, c in zip(bars, counts):
    a3.text(b_.get_x() + b_.get_width() / 2, c + .5, str(c), ha="center")
a3.set_ylabel("Unique studies"); a3.set_ylim(0, max(counts) + 6)
a3.set_title("(d) Evidence categories of 97 unique studies")
a3.text(0.98, 0.95, "Blue: primary set (n=%d)\nGrey: contextual studies" % int(uniq.eligible_primary.sum()),
        transform=a3.transAxes, ha="right", va="top", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "Figure1_quantitative_synthesis.png"), dpi=200)

# ---------- Figure 2 ----------
fig2, axf = plt.subplots(figsize=(7.2, 9.6)); axf.axis("off")
n_ent, n_dup, n_u = len(df), int((df.duplicate_of != "").sum()), len(uniq)
vc = uniq.screen_code.value_counts()
boxes = [
    (0.5, 0.93, "Identification\nPeer-reviewed Quantum-AI literature, 1 Jan 2024 \u2013 31 Dec 2025\nDatabase-level hit counts not retained (see Limitations)"),
    (0.5, 0.78, f"Unique studies in the evidence map (Tables 1\u20138)\nn = {n_u}"),
    (0.5, 0.63, "Classification by evidence category and quantum modality\n(title and tabulated method; Supplementary Data S1)"),
    (0.5, 0.46, f"Primary quantitative set\nquantum computational method (Q) = {vc.get('Q',0)} \u2022 quantum-inspired (QI) = {vc.get('QI',0)}\nclassical AI, quantum-oriented (QO) = {vc.get('QO',0)}"),
    (0.5, 0.28, f"Primary quantitative synthesis\nn = {int(uniq.eligible_primary.sum())} studies\nseparate metric families \u2022 bootstrap 95% CIs \u2022 Spearman \u03c1"),
    (0.5, 0.11, "Sensitivity analyses\nall unique studies \u2022 quantum methods only \u2022 bounded values removed"),
]
for x, y, t in boxes:
    axf.text(x, y, t, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.6", fc="#F6F8FA", ec="#5A6772", lw=1.4))
for y0, y1 in [(0.89, 0.82), (0.745, 0.665), (0.595, 0.51), (0.405, 0.325), (0.245, 0.145)]:
    axf.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="-|>", color="#5A6772", lw=1.4))
side = [
    (0.37, f"Contextual studies (evidence map only):\nclassical AI application (NQ) = {vc.get('NQ',0)}\nquantum method, no AI (NAI) = {vc.get('NAI',0)}\nreview, survey or policy (NE) = {vc.get('NE',0)}"),
]
for y, t in side:
    axf.text(0.985, y, t, ha="right", va="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.5", fc="#FFFFFF", ec="#999999", lw=1, ls="--"))
fig2.savefig(os.path.join(outdir, "Figure2_selection_flow.png"), dpi=200, bbox_inches="tight")
print("figures written")
