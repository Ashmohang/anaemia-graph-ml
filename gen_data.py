"""Synthetic household + individual blood-biomarker data for anaemia aetiology.

Structure (everything is planted, so models can be checked against ground truth):
  village -> household -> person
Household-level: consumption/diet, food insecurity, WASH, unobserved household effect,
  and *founder* haemoglobinopathy carrier status (inherited by children, so carriers cluster).
Person-level latent causes (multi-label, not mutually exclusive):
  iron deficiency, B12 deficiency, folate deficiency, inflammation,
  beta-thal trait, alpha-thal trait (silent/mild), HbS trait.
Observed: CBC for everyone; specialised panels are measured selectively (missing-at-random
  given the CBC, mimicking clinical triage), so many people lack ferritin / B12 / HPLC results.

Realism knobs: 35% of beta-thal carriers are borderline (HbA2 ~3.8%), iron/B12/folate deficiency depress HbA2,
HbS/beta-thal compound heterozygotes get HbS ~65% and extra anaemia. Not modelled: HbE, delta-thal, iron overload, CKD.
Replace with real PRAKASH data later: keep the column names in PERSON_FEATURES / LABELS.
"""
import numpy as np
import pandas as pd
from pathlib import Path

LABELS = ["iron", "b12", "folate", "inflam", "bthal", "athal", "hbs"]


def sig(x):
    return 1 / (1 + np.exp(-x))


def hb_threshold(age, female, pregnant):
    """WHO anaemia thresholds (g/dL)."""
    thr = np.where(age < 12, 11.5, np.where(age < 15, 12.0, np.where(female, 12.0, 13.0)))
    return np.where(pregnant, 11.0, thr)


def generate(n_hh=1500, n_vill=25, seed=0):
    rng = np.random.default_rng(seed)
    vill_eff = rng.normal(0, 0.5, n_vill)  # latent village burden (malaria, soil, water)
    vill_p = {
        "hbs": rng.beta(0.6, 6, n_vill),  # clustered HbS prevalence (tribal belts)
        "bthal": rng.beta(0.5, 12, n_vill),
        "athal": rng.beta(0.6, 7, n_vill),
    }
    bg = {"hbs": 0.01, "bthal": 0.015, "athal": 0.03}  # background carrier rate in married-in parent

    hh_rows, rows = [], []
    for h in range(n_hh):
        v = int(rng.integers(n_vill))
        income = rng.normal(0, 1)
        fi = int(rng.random() < sig(-1.0 * income - 0.3))
        veg = int(rng.random() < 0.35)
        flesh = 0 if veg else int(rng.poisson(max(0.3, 1.5 + 0.7 * income)))
        leafy = int(rng.poisson(max(0.3, 3.0 - 0.8 * fi)))
        dairy = int(rng.poisson(max(0.3, 3.0 + 0.8 * income + 0.5 * veg)))
        tea = int(rng.poisson(2.0))
        fort = int(rng.random() < 0.3)
        san = int(rng.random() < sig(0.9 * income - 0.2))
        wat = int(rng.random() < sig(0.7 * income + 0.4))
        hh_re = rng.normal(0, 0.4)  # unobserved household effect on iron status
        carrier_parent = {
            c: (rng.choice(["woman", "man"]) if rng.random() < vill_p[c][v] else None) for c in vill_p
        }

        members = [("woman", rng.uniform(20, 42)), ("man", rng.uniform(24, 55))]
        for _ in range(int(rng.integers(1, 5))):
            members.append(("kid", rng.uniform(6, 17.99)))

        hh_rows.append(dict(hh_id=h, village=v, income_z=income, food_insecure=fi, vegetarian=veg,
                            flesh_days=flesh, leafy_days=leafy, dairy_days=dairy, tea_cups=tea,
                            fortified=fort, sanitation=san, safe_water=wat, n_members=len(members)))

        parents_carry = {c: {r: (carrier_parent[c] == r) or (rng.random() < bg[c]) for r in ("woman", "man")}
                         for c in vill_p}
        for role, age in members:
            female = int(role == "woman" or (role == "kid" and rng.random() < 0.5))
            pregnant = int(role == "woman" and rng.random() < 0.10)
            geno = {}
            for c in vill_p:
                if role in ("woman", "man"):
                    geno[c] = int(parents_carry[c][role])
                else:  # inherit from each carrier parent with p=0.5
                    geno[c] = int(any(parents_carry[c][r] and rng.random() < 0.5 for r in ("woman", "man")))
            menstr = female and 12 <= age <= 49 and not pregnant
            hook = rng.random() < sig(-1.8 + 1.0 * (1 - san) + 0.3 * vill_eff[v])
            male = 1 - female
            iron = rng.random() < sig(-1.0 - 1.2 * male + 0.6 * menstr + 1.4 * pregnant + 0.4 * (age < 12)
                                      - 0.25 * (flesh - 2) - 0.12 * (leafy - 3) + 0.2 * (tea - 2) - 0.5 * fort
                                      + 0.5 * fi + 0.9 * hook + 0.4 * vill_eff[v] + hh_re - 0.2 * income)
            b12 = rng.random() < sig(-1.8 + 1.8 * veg - 0.25 * (dairy - 3) + 0.5 * (age >= 45))
            fol = rng.random() < sig(-1.8 - 0.3 * (leafy - 3) + 0.8 * pregnant + 0.5 * fi)
            inf = rng.random() < sig(-2.0 + 0.8 * (1 - san) + 0.5 * (1 - wat) + 0.4 * (age < 12) + vill_eff[v])
            rows.append(dict(hh_id=h, village=v, role=role, age=age, female=female, pregnant=pregnant,
                             iron=int(iron), b12=int(b12), folate=int(fol), inflam=int(inf),
                             bthal=geno["bthal"], athal=geno["athal"], hbs=geno["hbs"], hookworm=int(hook)))

    p = pd.DataFrame(rows)
    n = len(p)
    thr = hb_threshold(p.age.values, p.female.values, p.pregnant.values)
    d = {c: p[c].values.astype(bool) for c in LABELS}
    nrm = lambda m, s: rng.normal(m, s, n)

    # ---- haemoglobin ----
    hb = thr + 1.7 + nrm(0, 0.8)
    hb -= d["iron"] * rng.uniform(1.0, 2.5, n) + d["b12"] * rng.uniform(0.5, 1.5, n)
    hb -= d["folate"] * rng.uniform(0.4, 1.2, n) + d["inflam"] * rng.uniform(0.3, 1.0, n)
    hb -= d["bthal"] * nrm(1.3, 0.4) + d["athal"] * nrm(0.6, 0.3) + d["hbs"] * nrm(0.15, 0.2)
    compound = d["hbs"] & d["bthal"]  # HbS/beta-thal compound heterozygote: clinically significant disease
    hb -= compound * nrm(2.5, 0.8)
    hb = np.clip(hb, 4, 18)

    # ---- red cell indices ----
    mcv = np.where(p.age < 12, nrm(82, 4), nrm(88, 4)) + 3 * p.pregnant.values
    mcv -= d["iron"] * rng.uniform(6, 14, n) + d["athal"] * nrm(10, 3) + d["bthal"] * nrm(20, 3)
    mcv += (d["b12"] | d["folate"]) * rng.uniform(8, 18, n)
    mcv += nrm(0, 2.5)  # extra noise: more overlap between causes
    mcv = np.clip(mcv, 50, 125)
    rbc = hb * 30 / mcv  # Hct ~ 3*Hb => RBC = Hct*10/MCV
    mch = hb * 10 / rbc
    rdw = nrm(13.0, 0.7) + d["iron"] * nrm(2.5, 0.8) + (d["b12"] | d["folate"]) * nrm(2.0, 0.8) \
        + (d["bthal"] | d["athal"]) * nrm(1.0, 0.5) + d["inflam"] * 0.3

    # ---- iron / inflammation / vitamins ----
    ferr_mu = np.where(p.female == 0, np.log(100), np.where(p.age < 12, np.log(40), np.log(45)))
    lferr = rng.normal(ferr_mu, 0.55)
    lferr = np.where(d["iron"], rng.normal(np.log(12), 0.6, n), lferr)  # overlaps with replete tail
    lferr += d["inflam"] * rng.normal(0.9, 0.3, n)  # acute-phase rise masks iron deficiency
    lstfr = rng.normal(np.log(3.2), 0.25, n) + d["iron"] * np.log(2.2) + d["bthal"] * np.log(1.5) \
        + (d["b12"] | d["folate"]) * np.log(1.3)
    lcrp = np.where(d["inflam"], rng.normal(np.log(7), 0.7, n), rng.normal(np.log(1.2), 0.9, n))
    lb12 = np.where(d["b12"], rng.normal(np.log(140), 0.3, n), rng.normal(np.log(350), 0.35, n))
    lfol = np.where(d["folate"], rng.normal(np.log(2.8), 0.35, n), rng.normal(np.log(9), 0.4, n))

    # ---- haemoglobinopathy markers (HPLC) ----
    # 35% of beta-thal carriers have a mild/borderline phenotype (HbA2 ~3.8), the rest typical (~5.0)
    borderline = rng.random(n) < 0.35
    hba2 = np.where(d["bthal"], np.where(borderline, nrm(3.8, 0.3), nrm(5.0, 0.6)), nrm(2.75, 0.3))
    # deficiencies depress HbA2 (iron more when severe) -> mask beta-thal; plus assay noise
    hba2 -= d["iron"] * rng.uniform(0.2, 0.8, n) + (d["b12"] | d["folate"]) * rng.uniform(0.1, 0.6, n)
    hba2 += nrm(0, 0.15)
    hba2 = np.clip(hba2, 1.0, 8)
    hbf = np.clip(np.exp(nrm(np.log(0.5), 0.4)) + d["bthal"] * nrm(1.0, 0.6), 0, 10)
    hbs = np.where(d["hbs"], np.where(d["athal"], nrm(32, 3), nrm(37, 3)), 0.0)  # co-inherited alpha-thal lowers HbS%
    hbs = np.where(compound, nrm(65, 6), hbs)
    sol = np.where(hbs > 20, rng.random(n) < 0.97, rng.random(n) < 0.01).astype(float)

    anaem = hb < thr
    out = p.drop(columns=["hookworm"]).copy()
    out["compound_hbs_bthal"] = compound.astype(int)
    out["hb"], out["hb_threshold"], out["anaemic"] = hb, thr, anaem.astype(int)
    out["mcv"], out["rbc"], out["mch"], out["rdw"], out["mentzer"] = mcv, rbc, mch, rdw, mcv / rbc
    labs = {"ferritin": np.exp(lferr), "stfr": np.exp(lstfr), "crp": np.exp(lcrp), "b12_level": np.exp(lb12),
            "folate_level": np.exp(lfol), "hba2": hba2, "hbf": hbf, "hbs_pct": hbs, "sickle_sol": sol}
    # Selective testing (MAR given the CBC), like clinical triage
    avail = {
        "iron_panel": rng.random(n) < np.where(anaem, 0.75, 0.30),
        "vit_panel": rng.random(n) < np.where(mcv > 95, 0.70, 0.35),
        "hplc": rng.random(n) < np.where(mcv < 80, 0.75, 0.15),
        "sol": rng.random(n) < 0.5,
    }
    grp = {"ferritin": "iron_panel", "stfr": "iron_panel", "crp": "iron_panel", "b12_level": "vit_panel",
           "folate_level": "vit_panel", "hba2": "hplc", "hbf": "hplc", "hbs_pct": "hplc", "sickle_sol": "sol"}
    for k, v in labs.items():
        out[k] = np.where(avail[grp[k]], v, np.nan)
    return out, pd.DataFrame(hh_rows)


if __name__ == "__main__":
    here = Path(__file__).parent
    persons, hh = generate()
    persons.to_csv(here / "persons.csv", index=False)
    hh.to_csv(here / "households.csv", index=False)
    print(f"{len(hh)} households, {len(persons)} persons")
    print("anaemia prevalence:", round(persons.anaemic.mean(), 3))
    print("label prevalence:\n", persons[LABELS].mean().round(3).to_string())
    print("tested fraction:\n", persons[["ferritin", "b12_level", "hba2", "sickle_sol"]].notna().mean().round(2).to_string())
    a = persons[persons.anaemic == 1]
    print("among anaemic, any-cause share:\n", a[LABELS].mean().round(3).to_string())
