# Frozen Temporal Transformer classification accuracy

Calculated on 2026-11-10 using original saved predictions, not retraining or newly generated scores.

**Overall accuracy: 981/1186 = 82.715008431703%** (exact fraction 981/1186; decimal expansion is rounded at the displayed precision).

- Total evaluated: **1186 cases**, from **1146 unique patients**.
- Correct predictions: **981**.
- Incorrect predictions: **205**.
- True positives: **193**; true negatives: **788**; false positives: **107**; false negatives: **98**.

## Confusion matrix

Rows are actual labels; columns are predicted labels. Positive means prolonged postoperative stay.

| Actual / Predicted | 0 | 1 |
| --- | ---: | ---: |
| 0 | 788 (TN) | 107 (FP) |
| 1 | 98 (FN) | 193 (TP) |

## Metrics

| Metric | Value | Percentage |
| --- | ---: | ---: |
| Accuracy | 0.827150084317 | 82.715008432% |
| Precision | 0.643333333333 | 64.333333333% |
| Recall | 0.663230240550 | 66.323024055% |
| Sensitivity | 0.663230240550 | 66.323024055% |
| Specificity | 0.880446927374 | 88.044692737% |
| F1 | 0.653130287648 | 65.313028765% |
| Balanced Accuracy | 0.771838583962 | 77.183858396% |
| Auroc | 0.863925204938 | 86.392520494% |
| Auprc | 0.705996179964 | 70.599617996% |

Accuracy = (TP+TN)/N; precision = TP/(TP+FP); sensitivity/recall = TP/(TP+FN); specificity = TN/(TN+FP); F1 = 2TP/(2TP+FP+FN); balanced accuracy = (sensitivity+specificity)/2. AUROC uses continuous original probabilities; AUPRC is scikit-learn average precision, matching the original experiment. These ranking metrics do not use the binary cutoff.

## Frozen artifacts and validation

Checkpoint: `models/checkpoints/full_20261110/transformer.pt`.

Checkpoint SHA-256: `a10087ad745fd0a6d9a5c63097f4ddbaae524fadbc3b6d75399b4507e69486ec`.

Saved split and labels: `data/processed/full_20261110/dataset.pt`, `splits.test`, `cohort`, and `y`.

Dataset SHA-256: `46fb7d80002482df9ddcbef3788b8d3daa68c006c5a2fedf82161da54fa8f230`; matches the checkpoint's recorded training dataset hash.

Normalization: `data/processed/full_20261110/preprocessing.json`; statistics agree exactly with checkpoint and dataset. No preprocessing was refitted or inference needed because original scores exist.

Authoritative scores: `results/runs/full_20261110/errors/transformer_case_predictions.csv`. All 1,186 unique case IDs match the frozen test split exactly, with no extra, missing or duplicated case. Case-aligned probabilities exactly equal `results/statistical_validation/advanced_20261110/predictions_verified/transformer_test.csv`, which is the previously frozen-model-verified archive.

For every case, metadata in `data/metadata/cases.csv` was independently used to recompute `(dis - opend)/86400`. All intervals are finite and nonnegative; all cases are in-hospital survivors. These values agree with the saved cohort and prediction-file postoperative stays within 1e-12 days. Source metadata patient identifiers match prediction-file identifiers; no test patient occurs in training or validation.

The outcome label is `postoperative_stay_days > 7.396099537037037` days, the full-precision saved training-only 75th percentile (displayed as 7.39610). Recomputing the 75th percentile from training cases only reproduces the saved threshold. All recomputed labels match both frozen dataset labels and archived actual labels. No outcomes were used to alter scores or thresholds.

Binary prediction is `probability >= 0.36999999999999994` (displayed cutoff 0.370). Using literal 0.370 produces exactly the same labels for all cases. All newly calculated labels match archived predicted labels. There are 291 positive and 895 negative actual outcomes.

## Agreement with existing research results

Every requested metric present in `results/runs/full_20261110/metrics/transformer.json` agrees within absolute tolerance 1e-12, and its confusion matrix agrees exactly: `[[788, 107], [98, 193]]`. The archived accuracy is 0.8271500843170321, consistent with 981/1186.

`results/RESEARCH_VALIDATION_REPORT.md` identifies the same frozen experiment, test cases and training-only outcome definition, documents archived-score reproduction and refers to the passed `cohort_integrity_verified.json`. Its full-feature Transformer row rounds AUROC/AUPRC/F1 to 0.8639/0.7060/0.6531, agreeing with these calculations. `results/TRANSFORMER_FINAL_REPORT.md` additionally rounds sensitivity/specificity to 0.6632/0.8804, also agreeing. The narrative research report does not print every classification metric; the full-precision source experiment JSON supplies those comparisons. **No discrepancy was found.**

The passed earlier audit records Transformer score alignment and test-metric reproduction. This calculation reuses those original predictions as requested; it does not claim a new checkpoint forward pass was performed.

## Preservation and interpretation

All nine read source artifacts were hashed before and after calculation and remain unchanged. No checkpoint, split, preprocessing, threshold, prediction archive or existing research report was modified. Only this requested report was created.

This is case-level accuracy on an already examined internal retrospective held-out cohort, not external or clinical validation. Repeated operations mean 1,186 cases represent 1,146 patients. Excluding deaths makes the prediction population survivor-only. Overall accuracy depends on prevalence: 75.463744% of cases have label 0. Accuracy is distinct from an individual patient's model probability.
