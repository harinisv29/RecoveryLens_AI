# RecoveryLens AI — Research Validation Report

Analysis version: `advanced_20261110`. Frozen source experiment: `full_20261110`. This is **internal retrospective research evaluation**, not clinical validation.

## 1. Abstract

Re-auditing all 6,388 VitalDB cases confirmed 6,119 valid survivor cases and patient-separated splits of 3,981/952/1,186. Frozen-model predictions were verified against archived scores. Paired patient bootstrap analysis found the Transformer–GRU AUROC difference to be 0.0059, with a 95% interval spanning zero. Neural-model validation cross-fitting selected identity calibration; XGBoost selected Platt scaling with a slightly worse test Brier score. Nine new Transformer ablations and sampled attribution stability/completeness checks were executed. No original checkpoint was replaced.

## 2. Clinical motivation

Postoperative length of stay is a hospital-utilization and recovery-related endpoint. It is also influenced by surgery complexity, comorbidities, discharge practices and nonphysiological constraints. These experiments investigate associations and computational prediction, without asserting treatment benefit or clinical readiness.

## 3. Dataset description

VitalDB is a single-source retrospective surgical dataset. This project uses seven configured numeric physiological channels: HR, SpO2, SBP, DBP, MAP, EtCO2, and RR. Raw files are public deidentified recordings; unrelated high-frequency waveforms are not model inputs. There are 31,082 verified numeric track files. Official field definitions: https://vitaldb.net/dataset/.

## 4. Cohort selection

| Stage | Cases |
| --- | --- |
| Original cases | 6388 |
| Excluded deaths OR invalid outcomes | 82 |
| Outcome-eligible survivors | 6306 |
| With at least one downloaded configured track | 6305 |
| No configured tracks | 1 |
| With only one or two tracks | 126 |
| At least three tracks | 6179 |
| Fail signal-quality policy | 60 |
| Final eligible cases | 6119 |

Exclusions overlap: {'negative_postoperative_interval': 27, 'missing_or_nonfinite_discharge': 0, 'in_hospital_death': 57, 'invalid_operation_timing': 0, 'unknown_mortality_status': 0}. Two cases have both a negative interval and in-hospital death, so the 27 negative intervals plus 57 deaths yield 82 unique exclusions. There are no missing/nonfinite discharge timestamps. Of 6,306 outcome-eligible survivors, 6,305 have at least one configured track, 6,179 have at least three tracks, and 60 fail the fixed signal-quality policy.

Survivor-only: known in-hospital survivors. Excluding deaths changes the estimand; not an all-patient clinical outcome model.

There are 5854 unique included patients, 213 patients with multiple operations, 265 extra cases, and a maximum of 14 operations for one patient. No patient crosses train/validation/test. All initially examined patients are restricted to expanded training.

| Split | Cases | Patients | Prolonged stays |
| --- | --- | --- | --- |
| train | 3981 | 3791 | 995 |
| val | 952 | 917 | 224 |
| test | 1186 | 1146 | 291 |

The first CPU verification attempt is retained in `cohort_audit.json`. CPU GRU scores differed from archived CUDA scores by at most 0.0001578033. Original CUDA torch 2.5.1 and batch size 16 reproduced the neural predictions. The authoritative passed audit is `cohort_integrity_verified.json`; exact archived test probabilities are used after frozen-model reproduction. No cohort correction or baseline retraining was necessary.

## 5. Outcome definition

Elapsed postoperative stay is `(dis - opend) / 86400` days. VitalDB documents both fields as seconds relative to recording start. Negative intervals, invalid/missing outcome endpoints and in-hospital deaths are excluded. Prolonged stay is strictly greater than the **training-only 75th percentile: 7.3960995370 days**. This is a statistical threshold, not a clinically established standard. Source EMR dates cannot be independently verified.

## 6. Model architecture

Frozen Transformer: seven standardized physiology channels and seven observation indicators; 64-dimensional embedding, sinusoidal positions, two encoder layers, four attention heads, masked pooling, and a logit head. GRU uses packed sequences. XGBoost uses mean/std/min/max of normalized signals and missingness flags. Minute medians, broad artifact limits, train-only normalization, explicit padding masks, and a 720-minute cap are unchanged. Artifact limits are applied after minute aggregation and may miss brief within-minute artifacts.

## 7. Experimental methodology

All analyses require the passed cohort audit. Models use identical held-out cases and patient identifiers. Original checkpoints, datasets, preprocessing, app files and experimental results are hash-protected and were not overwritten. The holdout was already examined during the prior experiment; this secondary evaluation is exploratory, not a fresh independent validation. Bootstrap intervals condition on fitted models and fixed outcome labels. No analysis here establishes external or prospective validity.

## 8. Statistical comparison

Paired 2,000-resample patient-cluster bootstrap (seed 20261110). Every model uses the same sampled patients in each replicate. All 2,000 resamples contained both outcome classes. Positive differences favor the first named model.

| Comparison | AUROC difference | AUROC 95% CI | AUPRC difference | AUPRC 95% CI |
| --- | --- | --- | --- | --- |
| transformer minus gru | 0.0059 | [-0.0041, 0.0167] | -0.0078 | [-0.0306, 0.0171] |
| transformer minus xgboost | 0.0657 | [0.0462, 0.0848] | 0.1218 | [0.0828, 0.1584] |
| gru minus xgboost | 0.0597 | [0.0394, 0.0795] | 0.1296 | [0.0885, 0.1701] |

Transformer–GRU intervals include zero for both metrics; a superiority claim is unsupported. Neural-versus-XGBoost marginal intervals exclude zero, providing conditional exploratory evidence of a discrimination difference. Intervals are not multiplicity-adjusted and cannot establish clinical superiority. Bootstrap distributions are saved in `paired/bootstrap_distributions.csv`.

## 9. Calibration

Calibration methods were chosen by five-fold patient-group cross-fitted **validation** Brier score among identity, Platt scaling on clipped logits, and isotonic regression. The selected method was refitted using all validation cases only. Test outcomes were never used to fit or choose deployable calibration transformations. Reusing the base-model validation set means this is not independent nested outer validation.

| Model | Selected method | Raw Brier | Adjusted Brier | Raw intercept | Raw slope | Raw ECE |
| --- | --- | --- | --- | --- | --- | --- |
| transformer | identity | 0.11744 | 0.11744 | -0.0231 | 0.9715 | 0.02984 |
| gru | identity | 0.11762 | 0.11762 | 0.0340 | 1.0749 | 0.01544 |
| xgboost | platt | 0.14395 | 0.14401 | -0.0245 | 1.1182 | 0.03871 |

Calibration intercept and slope come from joint diagnostic logistic regression on test predictions. Calibration-in-the-large is additionally estimated with slope fixed to one. These test regressions describe miscalibration and are **not deployed calibration models**. ECE uses ten fixed equal-width bins, including weights for populated bins. Ideal joint intercept/slope are zero/one; ECE depends on binning and sample size.

Identity was selected for both neural models: their outputs remain uncalibrated model scores, not clinically validated probabilities. Platt scaling selected for XGBoost worsened test Brier slightly; validation improvement does not guarantee test improvement. Raw/calibrated diagnostics, bin data, over/underprediction summaries, OOF scores and separate JSON calibration parameters are retained.

![Transformer reliability](statistical_validation/advanced_20261110/calibration/transformer_reliability.png)

## 10. Subgroup analysis

Available attributes: age, sex, surgery type, actual surgical duration, ASA, and emergency/elective status. Test-set missing values: {'age': 0, 'sex': 0, 'optype': 0, 'asa': 22, 'emop': 0}. The analysis produced 84 model/group cells; 15 are suppressed for insufficient support. Counts are always retained.

Performance metrics require at least 50 cases and ten events and ten non-events. Confidence intervals additionally require 100 cases, 20 events/non-events, and 30 patient clusters; they use 1,000 patient-cluster resamples. Subgroup comparisons are descriptive, unadjusted and potentially confounded; overlapping intervals do not prove equality, and differences do not establish fairness or causality.

| Transformer group | n | Events | AUROC | AUPRC | Sensitivity | Specificity | Raw Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age: 40-59 | 478 | 95 | 0.863 | 0.701 | 0.621 | 0.911 | 0.1010 |
| age: 60-74 | 427 | 115 | 0.863 | 0.705 | 0.661 | 0.869 | 0.1278 |
| age: 75+ | 149 | 46 | 0.842 | 0.719 | 0.652 | 0.835 | 0.1475 |
| age: <40 | 132 | 35 | 0.877 | 0.754 | 0.800 | 0.845 | 0.1095 |
| sex: F | 576 | 89 | 0.863 | 0.622 | 0.618 | 0.918 | 0.0888 |
| sex: M | 610 | 202 | 0.849 | 0.745 | 0.683 | 0.836 | 0.1445 |
| surgery_type: Biliary/Pancreas | 151 | 47 | 0.919 | 0.818 | 0.660 | 0.933 | 0.1154 |
| surgery_type: Breast | 85 | 4 | Withheld | Withheld | Withheld | Withheld | Withheld |
| surgery_type: Colorectal | 229 | 39 | 0.782 | 0.414 | 0.410 | 0.900 | 0.1226 |
| surgery_type: Hepatic | 50 | 18 | 0.705 | 0.644 | 0.667 | 0.656 | 0.2029 |
| surgery_type: Major resection | 122 | 14 | 0.829 | 0.467 | 0.643 | 0.870 | 0.0970 |
| surgery_type: Minor resection | 111 | 15 | 0.744 | 0.522 | 0.333 | 1.000 | 0.0885 |
| surgery_type: Others | 157 | 55 | 0.874 | 0.775 | 0.727 | 0.843 | 0.1379 |
| surgery_type: Stomach | 118 | 44 | 0.717 | 0.598 | 0.636 | 0.676 | 0.2051 |
| surgery_type: Thyroid | 56 | 0 | Withheld | Withheld | Withheld | Withheld | Withheld |
| surgery_type: Transplantation | 65 | 49 | 0.967 | 0.990 | 0.918 | 0.875 | 0.0986 |
| surgery_type: Vascular | 42 | 6 | Withheld | Withheld | Withheld | Withheld | Withheld |
| surgery_duration_minutes: 120-240 | 371 | 110 | 0.735 | 0.543 | 0.518 | 0.801 | 0.1775 |
| surgery_duration_minutes: 60-120 | 352 | 40 | 0.746 | 0.342 | 0.275 | 0.955 | 0.0901 |
| surgery_duration_minutes: <=60 | 288 | 16 | 0.828 | 0.341 | 0.188 | 0.985 | 0.0454 |
| surgery_duration_minutes: >240 | 175 | 125 | 0.763 | 0.874 | 0.976 | 0.260 | 0.1635 |
| asa: 1 | 323 | 42 | 0.846 | 0.538 | 0.476 | 0.929 | 0.0859 |
| asa: 2 | 698 | 165 | 0.858 | 0.686 | 0.703 | 0.869 | 0.1159 |
| asa: 3 | 141 | 79 | 0.818 | 0.848 | 0.658 | 0.774 | 0.2041 |
| asa: 4+ | 2 | 1 | Withheld | Withheld | Withheld | Withheld | Withheld |
| asa: Unknown | 22 | 4 | Withheld | Withheld | Withheld | Withheld | Withheld |
| urgency: Elective | 1040 | 219 | 0.859 | 0.652 | 0.667 | 0.884 | 0.1103 |
| urgency: Emergency | 146 | 72 | 0.866 | 0.880 | 0.653 | 0.838 | 0.1684 |

Complete three-model subgroup values, supported intervals and calibration diagnostics are in `subgroups/summary.json` and `subgroups/subgroups.csv`.

## 11. Ablation results

Nine separately retrained **Transformer-family** experiments: six feature sets at the full window, plus all-feature first-30/60/90-minute models. This is not a 6-by-4 factorial design and does not retrain all three model families. Every variant retains the same cohort, labels, and patient splits. Input embedding dimensions change with the features; the no-flags model has seven inputs instead of fourteen. All variants use the RTX 4070 SUPER, AdamW 1e-4, batch 16, seed 42, maximum 40 epochs and validation-loss early stopping.

| Variant | Max minutes | AUROC | AUPRC | F1 | GPU loop seconds | Epochs |
| --- | --- | --- | --- | --- | --- | --- |
| hr_only | 720 | 0.8497 | 0.6859 | 0.6372 | 25.87 | 9 |
| bp_only | 720 | 0.8274 | 0.6699 | 0.5932 | 54.24 | 19 |
| spo2_only | 720 | 0.8286 | 0.6503 | 0.5881 | 68.30 | 24 |
| hr_bp_spo2 | 720 | 0.8508 | 0.6944 | 0.6312 | 74.48 | 26 |
| all_features | 720 | 0.8639 | 0.7060 | 0.6531 | 51.87 | 18 |
| no_missingness_flags | 720 | 0.8612 | 0.6991 | 0.6589 | 62.73 | 22 |
| first_30 | 30 | 0.7737 | 0.5376 | 0.5302 | 54.52 | 20 |
| first_60 | 60 | 0.7866 | 0.5332 | 0.5452 | 55.00 | 20 |
| first_90 | 90 | 0.8131 | 0.5759 | 0.5537 | 56.24 | 20 |

Early models use only `raw_sequence[:30/60/90]` and refit normalization using the same allowed training prefixes, so no later measurements enter early inputs or scaling. Surgeries shorter than the cap remain included; these are up-to-prefix predictions, not a strict ongoing-surgery landmark cohort. Full-window models represent end-of-recording prediction, capped at 720 minutes. Surgery duration/recording length remains visible through masks and positions, even in one-signal models. Feature-set differences do not isolate physiological causation. Removing flags still leaves implicit missingness through mean imputation. Every variant has a separate new checkpoint and retained validation/test results; no ablation model is promoted.

## 12. Explainability

Training-mean physiology, fixed observed/missingness indicators. Conditional physiology attribution; missingness contribution is deliberately not measured.

Uncalibrated Transformer logit, not probability or calibrated output

12 seeded held-out cases, 3 each TP/FP/FN/TN; descriptive, not exhaustive

Three seeded Gaussian perturbations per case: SD HR 1 bpm, SpO2 .2 percentage points, BP 2 mmHg, EtCO2/RR .5; observed entries only, clipped to configured validity limits

| Check | Computed summary |
| --- | --- |
| median_relative_completeness_32 | 0.000205 |
| median_relative_completeness_128 | 0.000048 |
| median_relative_completeness_256 | 0.000036 |
| median_feature_spearman | 1.000000 |
| median_temporal_spearman | 0.970251 |
| all_padding_checks_pass | True |

The existing 32-step implementation is checked against a direct trapezoidal reimplementation, the pre-sigmoid attribution target, exact checkpoint normalization, observed-value masks, minute alignment and masked padding. Completeness compares the attribution sum with the logit difference from the fixed-missingness baseline. Numerical convergence is assessed at 32/128/256 steps; rank stability uses three small perturbations per sampled case. Finite-step completeness errors and unstable cases must be interpreted, not hidden. Neither the baseline nor attribution establishes clinical causality. The original working explanation implementation remains unchanged.

## 13. Error analysis

| Model | TP | FP | FN | TN | FP >50% missing | FN >50% missing |
| --- | --- | --- | --- | --- | --- | --- |
| transformer | 193 | 107 | 98 | 788 | 16 | 30 |
| gru | 203 | 121 | 88 | 774 | 18 | 23 |
| xgboost | 184 | 189 | 107 | 706 | 4 | 37 |

Per-case false-positive and false-negative files include channel medians, physiological variability IQR, consecutive-minute absolute changes, observed fractions, HR>100 / MAP<65 / SpO2<94 fractions, surgery categories and postoperative-stay ranges. These thresholds are descriptive pattern markers, not treatment recommendations. Aggregates by TP/FP/FN/TN and probability-distribution/confusion-matrix plots are saved for all three models. Associations are unadjusted and cannot be interpreted as causes of postoperative stay or prediction error.

## 14. Limitations

- Retrospective single-source VitalDB evaluation, not external validation.
- Survivor-only selection excludes in-hospital deaths and changes the prediction population.
- Prolonged stay is a training-derived statistical upper-quartile endpoint, not an established clinical cutoff.
- The same holdout was previously examined; these are secondary exploratory analyses, not a new independent test.
- No prospective clinical validation or deployment readiness has been established.
- Surgical complexity, duration, case mix and hospital discharge practices may confound associations.
- Bootstrap uncertainty is conditional on fitted models and fixed labels; single-seed ablations do not measure training variability.
- Multiple exploratory comparisons and subgroup analyses are unadjusted for multiplicity.
- Recordings are capped at 720 minutes; transient within-minute artifacts may survive aggregation.
- Attribution does not establish causality; fixed missingness is not itself attributed.

## 15. Future work

External multicenter replication, prospective silent evaluation, predeclared clinical decision thresholds, competing-risk or mortality-inclusive endpoints, nested calibration validation, multiple-seed/repeated-split ablations, strict early landmarks, clinical artifact review, adjustment for procedure/discharge confounding, and independently evaluated deployment controls remain necessary.

## 16. Conclusion

The original 6,119-case survivor cohort and frozen predictions are verified. Transformer–GRU uncertainty does not support statistical superiority. Calibration selection retained the neural-model scores; the research dashboard explicitly labels them uncalibrated. The completed analyses are internal retrospective research, not clinical deployment validation. Final automated verification: **28 passed in 7.53s**. Version-asserted browser checks verified the original CSV workflow, calibration disclosure, charts, subgroup controls, explainability, mobile layout and augmented PDF. All 74 protected original files remain unchanged.

## Artifacts and launch

The research dashboard is an additive entry point: `python research_app.py` (same localhost port 5050). The original `app.py`, templates, static files, active checkpoint and existing experiment artifacts remain unchanged. See `research/README.md` for reproduction and preservation details.

Primary evidence directory: `results/statistical_validation/advanced_20261110/`. Calibration configurations: `models/calibration/advanced_20261110/`. Ablation checkpoints: `models/checkpoints/advanced_20261110/ablations/`.

## Method references

- VitalDB field dictionary and dataset: https://vitaldb.net/dataset/
- Probability calibration documentation: https://scikit-learn.org/stable/modules/calibration.html
- Sundararajan, Taly, Yan. Axiomatic Attribution for Deep Networks: https://arxiv.org/abs/1703.01365
