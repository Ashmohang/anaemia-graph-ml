"""Make figures + tables for the README from persons.csv and results.json."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

here = Path(__file__).parent
fig_dir = here / "figures"; fig_dir.mkdir(exist_ok=True)
P = pd.read_csv(here / "persons.csv")
R = json.loads((here / "results.json").read_text())
L = R["labels"]
NAMES = {"iron": "Iron def.", "b12": "B12 def.", "folate": "Folate def.", "inflam": "Inflammation",
         "bthal": "β-thal trait", "athal": "α-thal trait", "hbs": "HbS trait"}
COL = ["#9aa5b1", "#5b7db1", "#c9a227", "#b5413a"]
SHORT = ["Tabular", "Tabular + household context", "GNN (no edges)", "Hetero GNN"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

# Fig 1: anaemia is multifactorial
a = P[P.anaemic == 1]
ncause = a[["iron", "b12", "folate", "inflam", "bthal", "athal", "hbs"]].sum(axis=1).clip(upper=4)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
prev = a[L].mean().sort_values()
ax[0].barh([NAMES[k] for k in prev.index], prev.values * 100, color="#5b7db1")
ax[0].set_xlabel("% of anaemic people"); ax[0].set_title("Causes present among anaemic")
vc = ncause.value_counts(normalize=True).sort_index() * 100
ax[1].bar([str(i) if i < 4 else "4+" for i in vc.index], vc.values, color="#b5413a")
ax[1].set_xlabel("number of concurrent causes"); ax[1].set_ylabel("% of anaemic"); ax[1].set_title("Most have more than one")
co = np.array([[((a[i] == 1) & (a[j] == 1)).sum() / max(a[i].sum(), 1) for j in L] for i in L])
im = ax[2].imshow(co, cmap="Blues", vmin=0, vmax=1)
ax[2].set_xticks(range(7)); ax[2].set_yticks(range(7))
ax[2].set_xticklabels([NAMES[k] for k in L], rotation=60, ha="right"); ax[2].set_yticklabels([NAMES[k] for k in L])
ax[2].set_title("P(column | row) among anaemic"); plt.colorbar(im, ax=ax[2], fraction=0.046)
plt.tight_layout(); plt.savefig(fig_dir / "fig1_multifactorial.png"); plt.close()

# Fig 2: performance by label, all test persons (AUPRC) + Fig 3 HPLC-missing
def grouped(key, metric, title, fname, labels):
    idx = [L.index(l) for l in labels]
    fig, ax = plt.subplots(figsize=(10, 3.8)); w = 0.2
    for m, meth in enumerate(R["methods"]):
        arr = np.array(R[key][meth])[:, :, metric][:, idx]
        ax.bar(np.arange(len(idx)) + (m - 1.5) * w, arr.mean(0), w, yerr=arr.std(0), color=COL[m], label=SHORT[m], capsize=2)
    ax.set_xticks(range(len(idx))); ax.set_xticklabels([NAMES[l] for l in labels])
    ax.set_ylabel("AUPRC" if metric else "AUROC"); ax.set_title(title); ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(fig_dir / fname); plt.close()

grouped("all", 1, "Aetiology prediction, all test persons (mean ± sd over 3 splits)", "fig2_auprc_all.png", L)
grouped("no_hplc", 1, "Haemoglobinopathies when the person has NO HPLC result", "fig3_auprc_no_hplc.png", ["bthal", "athal", "hbs"])

# Fig 4: HbA2 masking
t = P[P.hba2.notna()]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
bins = np.linspace(1.5, 7, 45)
for k, (title, sel) in enumerate([("β-thal carriers, iron-replete", (t.bthal == 1) & (t.iron == 0)),
                                  ("β-thal carriers, iron-deficient", (t.bthal == 1) & (t.iron == 1))]):
    ax[k].hist(t[t.bthal == 0].hba2, bins, density=True, alpha=.5, color="#9aa5b1", label="non-carriers")
    ax[k].hist(t[sel].hba2, bins, density=True, alpha=.7, color="#b5413a", label=title)
    ax[k].axvline(3.5, ls="--", c="k", lw=1); ax[k].set_xlabel("HbA2 (%)"); ax[k].legend(frameon=False, fontsize=8)
    miss = (t[sel].hba2 <= 3.5).mean()
    ax[k].set_title(f"{title}: {miss:.0%} below the 3.5% cut-off")
ax[0].set_ylabel("density"); plt.tight_layout(); plt.savefig(fig_dir / "fig4_hba2_masking.png"); plt.close()

# Fig 5: why a graph helps - carriers cluster in households
g = P.groupby("hh_id")
rows = []
for c, nm in [("hbs", "HbS trait"), ("bthal", "β-thal trait")]:
    tot, cnt = g[c].transform("sum"), g[c].transform("count")
    other = (tot - P[c]) > 0
    rows.append((nm, P[c][other].mean(), P[c][~other].mean()))
fig, ax = plt.subplots(figsize=(5.2, 3.6)); x = np.arange(len(rows))
ax.bar(x - .18, [r[1] * 100 for r in rows], .36, color="#b5413a", label="another household member is a carrier")
ax.bar(x + .18, [r[2] * 100 for r in rows], .36, color="#9aa5b1", label="no other carrier in household")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylabel("% who are carriers")
ax.legend(frameon=False, fontsize=8); ax.set_title("Inherited conditions cluster in households")
plt.tight_layout(); plt.savefig(fig_dir / "fig5_household_clustering.png"); plt.close()

# Fig 0: graph schematic
fig, ax = plt.subplots(figsize=(8, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
def box(x, y, txt, col, w=1.7):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - .35), w, .7, boxstyle="round,pad=0.05", fc=col, ec="k", lw=.8))
    ax.text(x, y, txt, ha="center", va="center", fontsize=8)
box(5, 3.4, "Village\n(regional burden)", "#e8dcc4")
for x, h in [(2.5, "Household A"), (7.5, "Household B")]:
    box(x, 2.1, h + "\ndiet · income · WASH", "#cfe0cf", 2.3); ax.annotate("", (5, 3.05), (x, 2.45), arrowprops=dict(arrowstyle="-", lw=.8))
    for dx, lab in [(-1.3, "Mother\nCBC + HPLC"), (0, "Child 1\nCBC only"), (1.3, "Child 2\nCBC + ferritin")]:
        box(x + dx, .6, lab, "#c9d6ec", 1.2); ax.annotate("", (x, 1.75), (x + dx, .95), arrowprops=dict(arrowstyle="-", lw=.8))
ax.set_title("Heterogeneous graph: information flows between household members and up to the village", fontsize=9)
plt.tight_layout(); plt.savefig(fig_dir / "fig0_graph.png"); plt.close()

# Tables -> markdown
def md_table(key, metric):
    out = "| Method | " + " | ".join(NAMES[l] for l in L) + " | mean |\n|---|" + "---|" * 8 + "\n"
    for meth in R["methods"]:
        a_ = np.array(R[key][meth])[:, :, metric]
        out += f"| {meth} | " + " | ".join(f"{a_[:, j].mean():.2f}" for j in range(7)) + f" | **{a_.mean():.3f}** |\n"
    return out
rules = "| Screen | Sens | Spec | PPV | n+ |\n|---|---|---|---|---|\n" + "".join(
    f"| {r['name']} | {r['sens']:.2f} | {r['spec']:.2f} | {r['ppv']:.2f} | {r['n_pos']} |\n" for r in R["rules"])
(here / "tables.md").write_text("### AUPRC, all test persons\n\n" + md_table("all", 1) +
    "\n### AUPRC, persons with no HPLC result\n\n" + md_table("no_hplc", 1) +
    "\n### AUROC, all test persons\n\n" + md_table("all", 0) + "\n### Rule-based haemoglobinopathy screens\n\n" + rules)
print("figures:", sorted(p.name for p in fig_dir.iterdir()))
