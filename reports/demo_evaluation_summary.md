# Demo model evaluation report

## Dataset

- Source: bundled `demo_coding_practice.csv`.
- Records: 5,760; students: 60; topics: 8; weeks: 12.
- Nature: **synthetic / simulated**, not real student observations.

## Evaluation method

- Classification: fit on weeks 1–8, select on week 9, refit through week 9, test on weeks 10–12.
- Regression: predict the next observed week; select on week 9 and test forecast origins on weeks 10–11 (targets weeks 11–12).
- Classifier selection: highest validation accuracy, macro F1 as tie-breaker.
- Regressor selection: lowest validation MAE.
- Random Forest hyperparameters are optimized with a chronological `GridSearchCV` validation fold. The final test period is not used for tuning or model selection.

## Selected models and test metrics

- Classifier: **Random Forest (GridSearchCV)**.
- GridSearchCV parameters: `{'class_weight': 'balanced_subsample', 'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 200}`.
- Classification test accuracy: **77.99%**.
- Classification macro precision: **0.719**.
- Classification macro recall: **0.726**.
- Classification macro F1: **0.722**.
- Weak-class recall: **0.635**.
- Regressor: **Linear Regression**.
- Regression test MAE: **5.073 percentage points**.
- Regression test RMSE: **6.707**.
- Regression test R²: **0.871**.

## Comparison with the original notebook's reported results

| Metric | Original notebook | Upgraded pipeline |
|---|---:|---:|
| Classification accuracy | 76.46% | 77.99% |
| Weak-class recall | 53.00% | 63.46% |
| Regression MAE | 5.301 | 5.073 |
| Regression R² | 0.869 | 0.871 |

Classification accuracy changed by **+1.53 percentage points**. Regression MAE changed by **-0.228 points** (lower is better). The upgraded pipeline improves these metrics on this bundled simulated dataset, but this is not proof of better performance on real students.

## Important limitations

1. All reported results are based on simulated data. The generator creates patterns that are easier for ML models to learn than real-world student behaviour.
2. The time split evaluates later weeks for students represented in earlier weeks. It does not establish performance for completely unseen students.
3. Real-world accuracy can only be measured on representative, correctly labelled student practice records and a held-out evaluation period.
4. A high overall accuracy can hide weak performance on a minority class; macro precision, recall, F1 and the confusion matrix must be reported together.
