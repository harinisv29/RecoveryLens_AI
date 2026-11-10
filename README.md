# RecoveryLens AI## Public repository contentsThis repository includes application source, the frozen `full_20261110` Transformer checkpoint, saved normalization, and aggregate evaluation reports. Patient-level recordings, metadata, test predictions, sample recordings, and case-specific screenshots/PDFs remain local and are not distributed here. The sample-patient button requires a locally supplied `data/sample/real_recording.csv`; upload your own compatible recording for prediction. Research reproduction requires obtaining the source VitalDB data and regenerating the local dataset artifacts. Existing local files are preserved by the Git exclusions.For a fresh checkout, create a Python environment, install `requirements.txt`, and run `python transformer_app.py`. Open `http://127.0.0.1:5050`. The results describe internal retrospective research and do not establish clinical validation.

Temporal Transformer-Based Postoperative Risk Prediction. A local research prototype using real public VitalDB intraoperative numeric signals. **Not clinically validated and not for patient care.**

<!-- TRANSFORMER_ONLY_FINAL -->
## Final application: one live Temporal Transformer

Start the final hackathon application from this folder:

```powershell
.\.venv\Scripts\python.exe transformer_app.py
```

Open **http://127.0.0.1:5050**. Stop a previously running local server on that port before launching. The final entry point preserves the original app and research dashboard, but pins live inference to the validation-selected `models/checkpoints/full_20261110/transformer.pt`. It verifies the checkpoint SHA-256, feature schema, saved normalization, target and probability thresholds, and evaluation artifacts at startup. **The model loads once**, stays in evaluation mode, and uses `torch.inference_mode()` for scalar predictions. Integrated Gradients runs separately with input gradients enabled on the same cached model. No model is retrained or replaced.

Live architecture: CSV upload ? timestamp/numeric validation ? one-minute resampling/artifact checks ? saved training normalization and missingness indicators ? Temporal Transformer ? binary logit ? sigmoid ? dashboard, Integrated Gradients and PDF.

Verified checkpoint schema: **HR, SpO2, SBP, DBP, MAP, EtCO2, RR**, plus seven observation indicators (**14 inputs**). Uploads must have at least five minute bins and three channels with at least 20% observed coverage, matching the training cohort quality rule. Missing features/observations remain supported within that rule. Normalization is never refitted on uploads. Padding and attention masks remain separate from observation missingness. Maximum supported window is 720 minutes; this is the saved training cap.

- **Outcome threshold:** postoperative stay strictly greater than **7.39610 days**, derived from `(dis - opend) / 86400` and the training cohort's 75th percentile. In-hospital survivors only.
- **Binary probability threshold:** **0.370**, selected on validation data. This is distinct from the stay-duration threshold. No arbitrary low/medium/high clinical risk categories are introduced.
- **Displayed value:** estimated model probability, explicitly **not clinically calibrated**. Validation calibration selection retained identity for the Transformer.
- **Held-out metrics:** AUROC 0.8639, AUPRC 0.7060, F1 0.6531, sensitivity 0.6632, specificity 0.8804, verified against saved artifacts on 1,186 test cases.
- **Example:** `/api/example-csv` downloads the real deidentified VitalDB example in `data/sample/real_recording.csv`. The dashboard also provides a one-click real-example selector.
- **Research:** GRU and XGBoost remain baseline comparisons only. Transformer superiority over GRU was not statistically established. First-30/60/90-minute ablations remain retrospective findings and cannot be selected for live prediction.
- **Preservation:** datasets, original app/static files, checkpoints and research reports are retained. The previous README is preserved at `results/README.before_transformer_final.md`; the authorized documentation change is recorded in `results/transformer_final_preservation.json`.

The original `app.py` and `research_app.py` remain archival/legacy entry points. Use **`transformer_app.py`** for the final Transformer-only demonstration. Uploaded recordings are processed in memory. Research only; not clinically validated or ready for patient care.

<!-- CURRENT_FULL_RUN_START -->
## Current expanded experiment: full_20261110

All 6,306 outcome-eligible survivors were examined for the configured channels. **31,082 real numeric tracks** were verified (30,421 new; 661 reused), with 0 download failures. The final cohort contains **6,119 cases**; split sizes are {'train': 3981, 'val': 952, 'test': 1186} and positive counts are {'train': 995, 'val': 224, 'test': 291}. The training-only 75th percentile is **7.39609954 days**. Survivor-only population.

Selected application model: **transformer**, using validation AUROC with Brier tie-break. Checkpoint: `models\checkpoints\full_20261110\transformer.pt`. All original working checkpoints and the `initial_127` archive are retained.

| Model | AUROC (95% CI) | AUPRC (95% CI) | F1 | Sensitivity | Specificity | Brier | GPU seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | 0.8639 (0.8391–0.8871) | 0.7060 (0.6523–0.7573) | 0.6531 | 0.6632 | 0.8804 | 0.1174 | 52.19 |
| gru | 0.8580 (0.8311–0.8834) | 0.7138 (0.6624–0.7616) | 0.6602 | 0.6976 | 0.8648 | 0.1176 | 224.34 |
| xgboost | 0.7983 (0.7706–0.8261) | 0.5842 (0.5300–0.6399) | 0.5542 | 0.6323 | 0.7888 | 0.1439 | 0.38 |

Tests: ....................                                                     [100%]
20 passed in 6.52s. Edge browser verification passed CSV prediction, charts, explainability, mobile layout and PDF download. All 31,082 ledger entries matched SHA-256, with zero duplicate IDs. Raw verified bytes: 1,963,855,436.

Evidence, confusion matrices, calibration curves, bootstrap details, case-level errors and initial-versus-expanded comparisons: `results/runs/full_20261110/RESULTS.md` and `execution_summary.json`. Expanded normalization: `data/processed/full_20261110/preprocessing.json`. Cohort/exclusions: `data/metadata/full_20261110/cohort.json`.

Descriptive comparison, not a paired statistical improvement test. Cohorts, holdout patients and training-derived endpoint cutoffs differ; all initial subjects are restricted to expanded training.

No clinical validation or post-hoc calibration has been performed. Confidence intervals are conditional patient-cluster bootstrap intervals. GPU times measure training loops (neural loops include validation), excluding downloads, preprocessing and evaluation.

<!-- CURRENT_FULL_RUN_END -->

## Problem and endpoint

Estimate the probability of prolonged postoperative hospitalization from physiological trajectories recorded strictly during surgery (`opstart <= Time < opend`). No discharge, postoperative, demographic, or outcome-derived fields enter model inputs.

The downloaded public API schema **does not contain `los_postop`**, despite that field appearing in the official documentation. We derive elapsed postoperative stay as **`(dis - opend) / 86400` days**, using documented discharge and operation-end times relative to recording start. This is an elapsed interval, not calendar-day coding. All 6,388 cases have these endpoints, but 27 have negative intervals; those are invalid and excluded. In-hospital deaths are excluded because death truncates hospitalization, creating a survivor-cohort estimand. The audit found 6,306 nonfatal cases with valid outcome and operation timing before signal quality checks. See `data/metadata/outcome_audit.json`.

Prolonged means **strictly greater than the training split's 75th percentile**. This defines a statistical upper-tail task; it is not a validated clinical cutoff. The threshold must be re-estimated on a new training cohort. The binary decision cutoff maximizes validation F1. The application displays above/below that cutoff, without asserting clinically defined risk categories. These probabilities are uncalibrated model scores; a calibration plot and Brier score are provided, not proof of calibration.

## Dataset and provenance

- [Official dataset and field definitions](https://vitaldb.net/dataset/)
- [Official Python library](https://github.com/vitaldb/vitaldb-python)
- Public metadata: `https://api.vitaldb.net/cases`, track index: `https://api.vitaldb.net/trks`, numeric track: `https://api.vitaldb.net/{tid}`.
- Lee HC et al. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Scientific Data 9, 279 (2022). DOI: 10.1038/s41597-022-01411-5.

Respect the dataset's current license, attribution, and use conditions. Public single-center, non-cardiac surgery data is not representative of all patients or hospitals. No attempt is made to identify subjects.

## Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For GPU training install an appropriate CUDA-enabled PyTorch distribution using the official PyTorch installer instructions. Training automatically detects CUDA, uses autocast and gradient scaling on GPU, and runs on CPU otherwise. The GPU memory footprint is bounded by a batch size of 16 and a maximum 720 minute sequence; reduce batch size if necessary.

An optional `scripts/run_cuda.py --torch-site PATH SCRIPT [ARGS]` launcher can read a pre-existing compatible CUDA torch runtime without modifying it. All other dependencies remain in this project's environment. This session's execution details are recorded separately; this convenience path is not required on other computers.

## Reproduce the workflow

### Full-cohort expansion

The initial experiment is preserved under `results/runs/initial_127/`, including SHA-256 hashes in its backup manifest. The original working checkpoints are retained at their original paths. Expanded experiments use named directories under `data/processed/`, `data/metadata/`, `models/checkpoints/`, and `results/runs/`. Named training refuses to overwrite existing checkpoints or datasets.

```powershell
.\.venv\Scripts\python.exe scripts/backup_initial.py
.\.venv\Scripts\python.exe -u scripts/download_full_cohort.py --workers 16
.\.venv\Scripts\python.exe scripts/full_pipeline.py --run YOUR_UNIQUE_RUN --torch-site PATH_TO_CUDA_TORCH_SITE_PACKAGES
```

The full downloader requests all indexed tracks for the seven configured numeric channels in every outcome-eligible survivor. It reuses existing track-ID filenames, validates numeric schemas and finite timestamps/measurements, hashes completed files, retries transient errors, and records missing channels separately from failed network requests. It does not download unrelated high-frequency waveforms. Progress: `data/metadata/full_download_progress.json`; track outcomes and checksums: `full_download_tracks.jsonl`; failures: `full_download_failures.json`; missing tracks: `full_missing_channels.json`; case inventory: `full_case_inventory.json`. Resume by rerunning the downloader. Completed raw files are not fetched again.

The full pipeline waits for download completion, prepares a new cohort using the unchanged elapsed postoperative stay definition and survivor-only criteria, then trains all three models using the CUDA runtime. All patients seen during the initial experiment are restricted to expanded **training**. Fresh patient groups are split with seed 42 into approximately 64/16/20 training/validation/test; no subject crosses splits. The prolonged-stay endpoint is strictly greater than the new training subset's 75th percentile. Normalization uses only that same training subset. This protects the expanded validation and holdout sets from patients examined during prototype development.

Expanded evaluations include 2,000 **patient-cluster bootstrap** replicates with seed 42 for 95% percentile confidence intervals on AUROC and AUPRC. All observations for a sampled subject travel together; single-class bootstrap replicates are counted and skipped. Intervals condition on the fitted model and fixed endpoint; they do not capture training uncertainty. Calibration curves and Brier scores assess uncalibrated outputs. Confusion matrices, per-case probabilities and FP/FN classifications, and exploratory error summaries by duration and observed coverage are saved for each model.

The application selection is recorded in `models/active_model.json`, chosen by **validation AUROC**, with lowest validation Brier score as tie-breaker. Test performance never determines the selected model. Previous selection files are backed up. The app can serve Transformer, GRU, or XGBoost. Neural models use Integrated Gradients; XGBoost uses explicit per-minute/channel train-mean occlusion and labels it as nonadditive probability sensitivity, rather than as Integrated Gradients. To restore the initial Transformer, set the selection's `checkpoint` to `models/checkpoints/transformer.pt`. Restart an application process launched before these code changes.

Expanded-versus-initial comparisons are descriptive: the training-derived outcome cutoff and test patients differ, so score changes are not paired evidence of improvement. `results/runs/<run>/RESULTS.md` records actual counts, times, metrics, intervals, selection, and limitations. The workflow does not claim clinical validation.

Run from this folder:

```powershell
.\.venv\Scripts\python.exe scripts/download_dataset.py --limit 32
.\.venv\Scripts\python.exe scripts/inspect_dataset.py
.\.venv\Scripts\python.exe scripts/prepare_dataset.py
.\.venv\Scripts\python.exe scripts/generate_sample.py
.\.venv\Scripts\python.exe training/train.py --model transformer --epochs 40
.\.venv\Scripts\python.exe training/train.py --model gru --epochs 40
.\.venv\Scripts\python.exe training/train_baselines.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe app.py
```

The downloader resumes by validating existing CSV schemas, retries transient failures with exponential backoff, writes atomically, logs missing channels and failures, and records case-to-track mappings. `--limit 0` attempts all eligible cases. Increase the subset size if a fixed split lacks both classes; do not search for favorable test results. The default first-eligible-case sample is for engineering verification and is not a representative evaluation cohort.

## Preprocessing and architecture

Standard mapping: HR=`Solar8000/HR`, SpO2=`Solar8000/PLETH_SPO2`, SBP/DBP/MAP=`Solar8000/ART_SBP`, `ART_DBP`, `ART_MBP`, EtCO2=`Solar8000/ETCO2`, RR=`Solar8000/RR`. Missing invasive pressure or RR channels are retained as missing, never synthesized. Track timestamps share recording start as origin. Slice surgery first, then aggregate each elapsed minute with a median. Values outside broad configured physiological limits are discarded as artifacts; these limits need clinical review.

Training mean/std are fitted only on observed training values. Missing values are represented by the normalized training mean (zero) with a separate observation indicator. No interpolation across gaps is applied. Features never observed in training use mean zero and scale one and remain explicitly missing. Padding is separate from observation missingness, with attention masks. The maximum recording length is 720 minutes; longer operations are explicitly truncated by preprocessing, and uploads longer than 720 minutes are rejected.

Input `[batch, minutes, 14]` contains seven normalized channels plus seven observation flags. Transformer: linear embedding, sinusoidal positional encoding, two 64-dimensional encoder layers, four heads, dropout 0.1, masked temporal mean pooling, and a single-logit head. AdamW, BCE loss, gradient clipping, early stopping on validation BCE, and fixed seed 42. Hyperparameters are CLI-configurable. Inference applies sigmoid.

## Evaluation

### Initial 127-case experiment (archived)

The completed experiment downloaded 128 real cases; 127 passed outcome and signal-quality checks (case 86 failed the prespecified requirement of at least five minute bins and at least three channels with 20% observed coverage). The split contains 80/21/26 cases and 20/4/6 positive labels. The training-only 75th percentile is **7.49489294 days**. This is a **survivor-only** prediction population. Whole-dataset exclusions: 27 negative intervals, 0 missing/nonfinite discharge timestamps, 57 deaths, 0 invalid operation intervals, and 0 missing/inconsistent admission timestamps; 82 unique exclusions because two deaths also have negative intervals. Full case-level reasons are in `data/metadata/outcome_exclusions.csv`.

| Model | Test AUROC | Test AUPRC | F1 | Sensitivity | Specificity | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Temporal Transformer | 0.8000 | 0.6982 | 0.6154 | 0.6667 | 0.8500 | 0.1288 |
| GRU | 0.5667 | 0.3165 | 0.0000 | 0.0000 | 0.9000 | 0.1855 |
| XGBoost | 0.7667 | 0.5494 | 0.6000 | 0.5000 | 0.9500 | 0.1611 |

Transformer and GRU each trained 40 epochs on the RTX 4070 SUPER, taking **4.01** and **6.16 seconds**, respectively, for the synchronized epoch loops including validation. These times exclude downloads, setup, preprocessing, inference, and evaluation. Training read the existing CUDA torch 2.5.1+cu121 runtime through `scripts/run_cuda.py --torch-site C:\Users\Admin\.vscode\CharacterStudio\.venv\Lib\site-packages`; no existing project files were modified. Standalone inference and tests use the new environment's CPU torch. The best Transformer checkpoint is `models/checkpoints/transformer.pt`; normalization and preprocessing settings are saved both inside the checkpoint and in `data/processed/preprocessing.json`.

**14 automated tests passed.** Edge browser verification also passed real-file upload, charts, correctly ordered attribution labels, temporal heatmap, PDF download, and mobile overflow checks with no JavaScript errors. Evidence: `results/browser_verification.json`, `results/plots/dashboard_desktop.png`, `results/plots/dashboard_mobile.png`, and `results/explanations/sample_report.pdf`. `results/explanations/real_prediction.json` stores the public sample's real inference response. Test synthetic fixtures are limited to software behavior and never supply model training outcomes.

At the time of the initial archived experiment, only the 128-case feasibility subset had been downloaded. The expanded experiment above supersedes that scope limit. Full numeric-track retrieval requires many thousands of requests and is available as a resumable separate job using `--limit 0`. The first-eligible-case sampling and 26-case test set severely limit interpretation. No clinical validation or post-hoc probability calibration has been performed.

Subject-group splits are approximately 64/16/20 percent training/validation/test, fixed before training. No subject appears in multiple splits. XGBoost uses mean/std/min/max of the same normalized sequences and observation flags; GRU uses packed variable-length sequences. All models use the identical saved cohort, labels, normalization, and splits. Best neural checkpoints use validation loss; XGBoost uses validation early stopping. Decision thresholds use validation F1 only. Each final trained model is evaluated once on the held-out test set per run.

`results/metrics/{transformer,gru,xgboost}.json` contain AUROC, AUPRC, accuracy, precision, recall, F1, confusion matrix, Brier score, and calibration bins. `results/plots/` contains ROC, precision-recall, calibration, and neural loss curves. Small test cohorts have high sampling uncertainty: metrics are feasibility measurements, not performance claims. See `results/execution_summary.json` for actual completed work and limits. Do not rerun model selection based on test scores.

## Application and CSV input

Open **http://127.0.0.1:5050** after launching `app.py`. It serves a responsive Flask/HTML/CSS/JavaScript dashboard with locally vendored Chart.js, upload, physiological charts, risk, Integrated Gradients, temporal heatmap, and PDF export. Optional Google Fonts fall back to system sans-serif offline.

```csv
time_min,HR,SpO2,SBP,DBP,MAP,EtCO2,RR
0,82,99,120,80,93,35,14
1,85,98,118,78,91,36,14
2,91,98,110,74,86,35,15
3,97,97,102,68,79,34,15
4,105,96,95,62,73,33,16
```

The above values are **synthetic format examples only**, never used for training. `data/sample/real_recording.csv` is exported from real downloaded surgical data, with provenance beside it. Require at least five rows, strictly increasing nonnegative numeric timestamps, at least one supported signal, no unknown columns or infinite values. Blank observations and absent feature columns are supported. Only upload the intraoperative window. A maximum 5 MB request is accepted, with no persistent upload storage. The development server binds localhost. Predictions and reports are not logged or cached by the application.

Endpoints: `GET /`, `GET /api/health`, `GET /api/model-status`, `POST /api/predict`, `POST /api/report`. Both POST routes accept multipart `file` containing a CSV; reports recompute the prediction rather than accepting client-supplied scores. Missing checkpoints return 503, invalid uploads 400, oversized requests 413, concurrent inference 429. No random or fallback predictions are served. Only load trusted local checkpoints: PyTorch serialization is not safe for arbitrary files.

Integrated Gradients integrates the model logit from training-mean normalized physiology while holding missingness flags fixed. It reports signed time-feature attributions and absolute feature totals. Missingness contributions are not attributed by this baseline. It is a model-sensitivity explanation, not causality, physiological mechanism, or clinical reasoning. PDF generation includes signals, attribution heatmap, endpoint, quality, prediction, and limitations, entirely in memory.

## Limitations and future work

Research infrastructure, not a deployable clinical product. Full-cohort training, external and prospective validation, uncertainty intervals, procedure-aware outcomes, competing-risk treatment of mortality, calibration fitting on separate validation data, drift/robustness audits, timestamp sensitivity, alternative signal sources, better quality gates, and clinical assessment of artifact limits remain necessary. Inference currently permits low-completeness inputs and may be out of distribution; inspect the quality summary. Deployment would additionally need authentication, TLS, rate limiting, audit controls, and a production server. The local Flask server is for development verification only.
