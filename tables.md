### AUPRC, all test persons

| Method | Iron def. | B12 def. | Folate def. | Inflammation | β-thal trait | α-thal trait | HbS trait | mean |
|---|---|---|---|---|---|---|---|---|
| Tabular (person + household covariates) | 0.99 | 0.92 | 0.87 | 0.74 | 0.82 | 0.40 | 0.78 | **0.788** |
| Tabular + household context (hand-engineered) | 0.99 | 0.92 | 0.86 | 0.73 | 0.91 | 0.44 | 0.88 | **0.819** |
| GNN, no message passing (ablation) | 0.99 | 0.89 | 0.85 | 0.72 | 0.82 | 0.42 | 0.77 | **0.782** |
| Heterogeneous GNN (person-household-village) | 0.99 | 0.92 | 0.87 | 0.74 | 0.83 | 0.42 | 0.83 | **0.798** |

### AUPRC, persons with no HPLC result

| Method | Iron def. | B12 def. | Folate def. | Inflammation | β-thal trait | α-thal trait | HbS trait | mean |
|---|---|---|---|---|---|---|---|---|
| Tabular (person + household covariates) | 0.98 | 0.93 | 0.88 | 0.75 | 0.20 | 0.29 | 0.65 | **0.669** |
| Tabular + household context (hand-engineered) | 0.98 | 0.93 | 0.88 | 0.73 | 0.51 | 0.34 | 0.80 | **0.739** |
| GNN, no message passing (ablation) | 0.98 | 0.91 | 0.87 | 0.73 | 0.32 | 0.32 | 0.64 | **0.680** |
| Heterogeneous GNN (person-household-village) | 0.98 | 0.93 | 0.88 | 0.74 | 0.32 | 0.35 | 0.72 | **0.703** |

### AUROC, all test persons

| Method | Iron def. | B12 def. | Folate def. | Inflammation | β-thal trait | α-thal trait | HbS trait | mean |
|---|---|---|---|---|---|---|---|---|
| Tabular (person + household covariates) | 0.99 | 0.97 | 0.96 | 0.87 | 0.97 | 0.83 | 0.94 | **0.932** |
| Tabular + household context (hand-engineered) | 0.99 | 0.97 | 0.95 | 0.87 | 0.99 | 0.83 | 0.97 | **0.940** |
| GNN, no message passing (ablation) | 0.99 | 0.96 | 0.95 | 0.86 | 0.97 | 0.84 | 0.93 | **0.928** |
| Heterogeneous GNN (person-household-village) | 0.99 | 0.97 | 0.96 | 0.86 | 0.98 | 0.86 | 0.96 | **0.940** |

### Rule-based haemoglobinopathy screens

| Screen | Sens | Spec | PPV | n+ |
|---|---|---|---|---|
| CBC only: Mentzer<13 & MCV<80 -> any thal trait (all persons) | 0.26 | 0.98 | 0.59 | 727 |
| CBC only: Mentzer<13 & MCV<80 -> beta-thal (all persons) | 0.44 | 0.96 | 0.25 | 183 |
| HPLC: HbA2>3.5% -> beta-thal (tested) | 0.91 | 1.00 | 0.94 | 131 |
| HPLC: HbA2>3.5% -> beta-thal (tested, iron-deficient only) | 0.88 | 1.00 | 1.00 | 40 |
| HPLC: HbA2>3.5% -> beta-thal (tested, iron-replete only) | 0.92 | 0.99 | 0.91 | 91 |
| HPLC: HbS>20% -> HbS trait (tested) | 1.00 | 1.00 | 1.00 | 123 |
| Solubility test -> HbS trait (tested) | 0.95 | 0.99 | 0.92 | 221 |
| alpha-thal: microcytosis (Mentzer<13, MCV<80) with normal HbA2 (tested) | 0.29 | 0.94 | 0.45 | 285 |
