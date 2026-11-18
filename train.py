"""Anaemia aetiology: tabular baselines vs heterogeneous GNN, + haemoglobinopathy evaluation.

Run: python gen_data.py && python train.py
Splits are by HOUSEHOLD (no family leakage). 3 seeds -> mean +- sd. Writes results.md.
"""
import warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

warnings.filterwarnings("ignore")
here = Path(__file__).parent
P = pd.read_csv(here / "persons.csv")
H = pd.read_csv(here / "households.csv")
LABELS = ["iron", "b12", "folate", "inflam", "bthal", "athal", "hbs"]
LABCOLS = ["ferritin", "stfr", "crp", "b12_level", "folate_level", "hba2", "hbf", "hbs_pct", "sickle_sol"]
LOGCOLS = ["ferritin", "stfr", "crp", "b12_level", "folate_level"]
HHF = ["income_z", "food_insecure", "vegetarian", "flesh_days", "leafy_days", "dairy_days", "tea_cups",
       "fortified", "sanitation", "safe_water", "n_members"]

P["hb_gap"] = P.hb - P.hb_threshold
for r in ["woman", "man", "kid"]:
    P[f"role_{r}"] = (P.role == r).astype(float)
for c in LOGCOLS:
    P[c] = np.log(P[c])
BASE = ["age", "female", "pregnant", "role_woman", "role_man", "role_kid", "hb", "hb_gap", "mcv", "mch", "rbc",
        "rdw", "mentzer"]
P = P.merge(H, on=["hh_id", "village"])


def household_context(df):
    """Hand-engineered leave-one-out household aggregates of the OTHER members (strong tabular baseline)."""
    cols = ["hb", "mcv", "mentzer"] + LABCOLS
    g = df.groupby("hh_id")
    out = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c].fillna(0)
        n = df[c].notna().astype(float)
        ssum, ncnt = g[c].transform(lambda x: x.fillna(0).sum()), g[c].transform(lambda x: x.notna().sum())
        cnt = ncnt - n
        out[f"hh_{c}"] = np.where(cnt > 0, (ssum - s) / cnt.replace(0, np.nan), np.nan)
    return out


CTX = household_context(P)
X_tab1 = P[BASE + LABCOLS + HHF]
X_tab2 = pd.concat([X_tab1, CTX], axis=1)
Y = P[LABELS].values.astype(float)


def metrics(y, s):
    return roc_auc_score(y, s), average_precision_score(y, s)


# ---------------- GNN ----------------
class Net(nn.Module):
    def __init__(self, dp, dh, nv, hid=64, out=len(LABELS), p=0.3, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        self.ep, self.eh, self.ev = nn.Linear(dp, hid), nn.Linear(dh, hid), nn.Embedding(nv, hid)
        mk = lambda: HeteroConv({
            ("person", "in", "household"): SAGEConv(hid, hid), ("household", "has", "person"): SAGEConv(hid, hid),
            ("household", "in", "village"): SAGEConv(hid, hid), ("village", "has", "household"): SAGEConv(hid, hid)})
        self.convs = nn.ModuleList([mk(), mk()])
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Dropout(p), nn.Linear(hid, out))

    def forward(self, d):
        x0 = {"person": torch.relu(self.ep(d["person"].x)), "household": torch.relu(self.eh(d["household"].x)),
              "village": self.ev(torch.arange(d["village"].num_nodes))}
        x = x0
        for conv in self.convs:
            x = conv(x, d.edge_index_dict)
            x = {k: self.drop(torch.relu(v)) + x0[k] for k, v in x.items()}
        return self.head(torch.cat([x["person"], x0["person"]], 1))


def build_graph():
    tr = P[BASE + LABCOLS].copy()
    mask = tr[LABCOLS].notna().astype(float).values
    d = HeteroData()
    d["person"].x_raw, d["person"].mask = tr.values.astype(np.float32), mask.astype(np.float32)
    d["household"].x_raw = H[HHF].values.astype(np.float32)
    d["village"].num_nodes = H.village.max() + 1
    pe = torch.tensor(np.vstack([np.arange(len(P)), P.hh_id.values]), dtype=torch.long)
    he = torch.tensor(np.vstack([H.hh_id.values, H.village.values]), dtype=torch.long)
    d["person", "in", "household"].edge_index = pe
    d["household", "has", "person"].edge_index = pe.flip(0)
    d["household", "in", "village"].edge_index = he
    d["village", "has", "household"].edge_index = he.flip(0)
    return d


G = build_graph()


def run_gnn(tr_idx, va_idx, te_idx, seed, use_graph=True):
    torch.manual_seed(seed)
    d = G.clone()
    xr = d["person"].x_raw
    mu, sd = np.nanmean(xr[tr_idx], 0), np.nanstd(xr[tr_idx], 0) + 1e-6
    xp = np.nan_to_num((xr - mu) / sd, nan=0.0)
    d["person"].x = torch.tensor(np.hstack([xp, d["person"].mask]), dtype=torch.float32)
    hx = d["household"].x_raw
    hmu, hsd = hx[P.hh_id.values[tr_idx]].mean(0), hx[P.hh_id.values[tr_idx]].std(0) + 1e-6
    d["household"].x = torch.tensor((hx - hmu) / hsd, dtype=torch.float32)
    if not use_graph:  # ablation: person MLP + household covariates, no message passing
        for et in d.edge_types:
            d[et].edge_index = torch.zeros(2, 0, dtype=torch.long)
    y = torch.tensor(Y, dtype=torch.float32)
    pos = y[tr_idx].mean(0)
    pw = ((1 - pos) / pos).sqrt().clamp(1, 5)
    model = Net(d["person"].x.shape[1], d["household"].x.shape[1], int(d["village"].num_nodes))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state, bad = -1, None, 0
    for ep in range(300):
        model.train(); opt.zero_grad()
        loss = lossf(model(d)[tr_idx], y[tr_idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(d)).numpy()
        v = np.mean([average_precision_score(Y[va_idx, j], s[va_idx, j]) for j in range(len(LABELS))])
        if v > best:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 40:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(d)).numpy()


def run_hgb(X, tr_idx, seed):
    S = np.zeros((len(P), len(LABELS)))
    for j in range(len(LABELS)):
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0,
                                           random_state=seed)
        m.fit(X.iloc[tr_idx], Y[tr_idx, j])
        S[:, j] = m.predict_proba(X)[:, 1]
    return S


# ---------------- experiment ----------------
no_hplc = P.hba2.isna().values
methods = ["Tabular (person + household covariates)", "Tabular + household context (hand-engineered)",
           "GNN, no message passing (ablation)", "Heterogeneous GNN (person-household-village)"]
res = {m: [] for m in methods}      # per seed: (auroc[7], auprc[7])
res_nohplc = {m: [] for m in methods}
for seed in range(3):
    rng = np.random.default_rng(100 + seed)
    hh = rng.permutation(H.hh_id.values)
    cut1, cut2 = int(0.6 * len(hh)), int(0.8 * len(hh))
    split = {h: 0 for h in hh[:cut1]} | {h: 1 for h in hh[cut1:cut2]} | {h: 2 for h in hh[cut2:]}
    sp = P.hh_id.map(split).values
    tr, va, te = [np.where(sp == k)[0] for k in range(3)]
    preds = {methods[0]: run_hgb(X_tab1, tr, seed), methods[1]: run_hgb(X_tab2, tr, seed),
             methods[2]: run_gnn(tr, va, te, seed, use_graph=False), methods[3]: run_gnn(tr, va, te, seed)}
    for m, S in preds.items():
        res[m].append(np.array([metrics(Y[te, j], S[te, j]) for j in range(len(LABELS))]))
        sub = te[no_hplc[te]]
        res_nohplc[m].append(np.array([metrics(Y[sub, j], S[sub, j]) for j in range(len(LABELS))]))
    print("seed", seed, "done", flush=True)

lines = [f"# Anaemia aetiology on synthetic data ({len(P)} persons, {len(H)} households, 3 household-level splits)\n"]
lines.append("Prevalence: " + ", ".join(f"{l} {P[l].mean():.1%}" for l in LABELS) + "\n")


def table(R, title, col):
    t = f"\n## {title}\n\n| Method | " + " | ".join(LABELS) + " | mean |\n|---|" + "---|" * (len(LABELS) + 1) + "\n"
    for m in methods:
        a = np.stack(R[m])[:, :, col]
        t += f"| {m} | " + " | ".join(f"{a[:, j].mean():.3f}±{a[:, j].std():.3f}" for j in range(len(LABELS))) \
            + f" | {a.mean():.3f} |\n"
    return t


lines += [table(res, "AUROC - all test persons", 0), table(res, "AUPRC - all test persons", 1),
          table(res_nohplc, "AUROC - test persons WITHOUT an HPLC result", 0),
          table(res_nohplc, "AUPRC - test persons WITHOUT an HPLC result", 1)]

# ---------------- haemoglobinopathy rule-based screening (no learning) ----------------
def sens_spec(y, pred):
    tp, fn = ((y == 1) & pred).sum(), ((y == 1) & ~pred).sum()
    tn, fp = ((y == 0) & ~pred).sum(), ((y == 0) & pred).sum()
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1), int(y.sum())


raw = pd.read_csv(here / "persons.csv")
thal = ((raw.bthal == 1) | (raw.athal == 1)).values
hp = raw.hba2.notna().values
rules = [
    ("CBC only: Mentzer<13 & MCV<80 -> any thal trait (all persons)", thal, ((raw.mentzer < 13) & (raw.mcv < 80)).values),
    ("CBC only: Mentzer<13 & MCV<80 -> beta-thal (all persons)", raw.bthal.values, ((raw.mentzer < 13) & (raw.mcv < 80)).values),
    ("HPLC: HbA2>3.5% -> beta-thal (tested)", raw.bthal.values[hp], (raw.hba2 > 3.5).values[hp]),
    ("HPLC: HbA2>3.5% -> beta-thal (tested, iron-deficient only)", raw.bthal.values[hp & (raw.iron == 1).values],
     (raw.hba2 > 3.5).values[hp & (raw.iron == 1).values]),
    ("HPLC: HbA2>3.5% -> beta-thal (tested, iron-replete only)", raw.bthal.values[hp & (raw.iron == 0).values],
     (raw.hba2 > 3.5).values[hp & (raw.iron == 0).values]),
    ("HPLC: HbS>20% -> HbS trait (tested)", raw.hbs.values[hp], (raw.hbs_pct > 20).values[hp]),
    ("Solubility test -> HbS trait (tested)", raw.hbs.values[raw.sickle_sol.notna().values],
     (raw.sickle_sol == 1).values[raw.sickle_sol.notna().values]),
    ("alpha-thal: microcytosis (Mentzer<13, MCV<80) with normal HbA2 (tested)", raw.athal.values[hp],
     (((raw.mentzer < 13) & (raw.mcv < 80)) & (raw.hba2 <= 3.5)).values[hp]),
]
lines.append("\n## Rule-based haemoglobinopathy screens (all data, no learning)\n\n| Screen | Sensitivity | Specificity | PPV | n positives |\n|---|---|---|---|---|\n")
for name, y, pr in rules:
    se, sp_, ppv, npos = sens_spec(np.asarray(y), np.asarray(pr))
    lines.append(f"| {name} | {se:.2f} | {sp_:.2f} | {ppv:.2f} | {npos} |\n")
import json
dump = {"labels": LABELS, "methods": methods,
        "all": {m: np.stack(res[m]).tolist() for m in methods},
        "no_hplc": {m: np.stack(res_nohplc[m]).tolist() for m in methods},
        "rules": [dict(name=n, sens=float(sens_spec(np.asarray(y), np.asarray(pr))[0]),
                       spec=float(sens_spec(np.asarray(y), np.asarray(pr))[1]),
                       ppv=float(sens_spec(np.asarray(y), np.asarray(pr))[2]),
                       n_pos=sens_spec(np.asarray(y), np.asarray(pr))[3]) for n, y, pr in rules]}
(here / "results.json").write_text(json.dumps(dump))
(here / "results.md").write_text("".join(lines))
print("".join(lines))
