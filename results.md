# Anaemia aetiology on synthetic data (6759 persons, 1500 households, 3 household-level splits)
Prevalence: iron 37.1%, b12 27.0%, folate 20.7%, inflam 24.0%, bthal 2.7%, athal 8.2%, hbs 6.1%

## AUROC - all test persons

| Method | iron | b12 | folate | inflam | bthal | athal | hbs | mean |
|---|---|---|---|---|---|---|---|---|
| Tabular (person + household covariates) | 0.993±0.000 | 0.969±0.003 | 0.955±0.002 | 0.872±0.014 | 0.968±0.024 | 0.828±0.032 | 0.937±0.018 | 0.932 |
| Tabular + household context (hand-engineered) | 0.994±0.000 | 0.967±0.003 | 0.954±0.001 | 0.865±0.010 | 0.994±0.001 | 0.833±0.037 | 0.973±0.006 | 0.940 |
| GNN, no message passing (ablation) | 0.993±0.001 | 0.959±0.004 | 0.952±0.001 | 0.856±0.009 | 0.973±0.014 | 0.837±0.010 | 0.926±0.026 | 0.928 |
| Heterogeneous GNN (person-household-village) | 0.992±0.001 | 0.966±0.004 | 0.958±0.004 | 0.865±0.010 | 0.984±0.003 | 0.857±0.018 | 0.959±0.014 | 0.940 |

## AUPRC - all test persons

| Method | iron | b12 | folate | inflam | bthal | athal | hbs | mean |
|---|---|---|---|---|---|---|---|---|
| Tabular (person + household covariates) | 0.989±0.001 | 0.922±0.011 | 0.866±0.007 | 0.743±0.019 | 0.817±0.052 | 0.400±0.031 | 0.782±0.051 | 0.788 |
| Tabular + household context (hand-engineered) | 0.990±0.000 | 0.919±0.011 | 0.865±0.009 | 0.727±0.013 | 0.907±0.011 | 0.441±0.036 | 0.883±0.035 | 0.819 |
| GNN, no message passing (ablation) | 0.988±0.002 | 0.895±0.015 | 0.853±0.005 | 0.723±0.017 | 0.824±0.033 | 0.422±0.040 | 0.768±0.054 | 0.782 |
| Heterogeneous GNN (person-household-village) | 0.988±0.002 | 0.916±0.017 | 0.868±0.011 | 0.737±0.021 | 0.831±0.031 | 0.416±0.054 | 0.827±0.042 | 0.798 |

## AUROC - test persons WITHOUT an HPLC result

| Method | iron | b12 | folate | inflam | bthal | athal | hbs | mean |
|---|---|---|---|---|---|---|---|---|
| Tabular (person + household covariates) | 0.992±0.001 | 0.967±0.004 | 0.954±0.002 | 0.877±0.013 | 0.896±0.067 | 0.810±0.043 | 0.875±0.029 | 0.910 |
| Tabular + household context (hand-engineered) | 0.992±0.001 | 0.965±0.004 | 0.952±0.000 | 0.869±0.010 | 0.981±0.006 | 0.812±0.043 | 0.949±0.009 | 0.931 |
| GNN, no message passing (ablation) | 0.991±0.001 | 0.957±0.006 | 0.949±0.000 | 0.866±0.007 | 0.908±0.042 | 0.824±0.013 | 0.861±0.044 | 0.908 |
| Heterogeneous GNN (person-household-village) | 0.991±0.001 | 0.965±0.007 | 0.954±0.004 | 0.869±0.008 | 0.950±0.009 | 0.855±0.028 | 0.928±0.020 | 0.930 |

## AUPRC - test persons WITHOUT an HPLC result

| Method | iron | b12 | folate | inflam | bthal | athal | hbs | mean |
|---|---|---|---|---|---|---|---|---|
| Tabular (person + household covariates) | 0.983±0.000 | 0.931±0.010 | 0.881±0.004 | 0.748±0.013 | 0.202±0.064 | 0.291±0.058 | 0.646±0.068 | 0.669 |
| Tabular + household context (hand-engineered) | 0.984±0.001 | 0.927±0.010 | 0.876±0.008 | 0.733±0.009 | 0.514±0.109 | 0.340±0.050 | 0.795±0.047 | 0.739 |
| GNN, no message passing (ablation) | 0.981±0.001 | 0.907±0.015 | 0.866±0.007 | 0.733±0.015 | 0.316±0.146 | 0.320±0.036 | 0.636±0.063 | 0.680 |
| Heterogeneous GNN (person-household-village) | 0.980±0.002 | 0.927±0.019 | 0.879±0.010 | 0.738±0.012 | 0.323±0.153 | 0.354±0.047 | 0.719±0.050 | 0.703 |

## Rule-based haemoglobinopathy screens (all data, no learning)

| Screen | Sensitivity | Specificity | PPV | n positives |
|---|---|---|---|---|
| CBC only: Mentzer<13 & MCV<80 -> any thal trait (all persons) | 0.26 | 0.98 | 0.59 | 727 |
| CBC only: Mentzer<13 & MCV<80 -> beta-thal (all persons) | 0.44 | 0.96 | 0.25 | 183 |
| HPLC: HbA2>3.5% -> beta-thal (tested) | 0.91 | 1.00 | 0.94 | 131 |
| HPLC: HbA2>3.5% -> beta-thal (tested, iron-deficient only) | 0.88 | 1.00 | 1.00 | 40 |
| HPLC: HbA2>3.5% -> beta-thal (tested, iron-replete only) | 0.92 | 0.99 | 0.91 | 91 |
| HPLC: HbS>20% -> HbS trait (tested) | 1.00 | 1.00 | 1.00 | 123 |
| Solubility test -> HbS trait (tested) | 0.95 | 0.99 | 0.92 | 221 |
| alpha-thal: microcytosis (Mentzer<13, MCV<80) with normal HbA2 (tested) | 0.29 | 0.94 | 0.45 | 285 |
