# Additive research evaluation

This package adds internal retrospective research evaluation to the existing RecoveryLens AI project. It does not replace the primary model or overwrite the original app, templates, static assets, datasets, checkpoints, or expanded experiment. The 6,119-case survivor cohort was independently rebuilt from raw tracks before advanced evaluation. Source inputs are the frozen `full_20261110` experiment.

The authoritative audit is `results/statistical_validation/advanced_20261110/cohort_integrity_verified.json`. The first CPU audit is retained unchanged: its cohort checks passed but its strict GRU score comparison failed because CPU outputs differed by up to 0.0001578033 from the original CUDA implementation. Reproduction using the original neural CUDA runtime and batch size 16 resolved the difference; archived test scores are used after verification. No cohort correction was necessary.

## Launch

Install the additional PDF dependency if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r research/requirements.txt
.\.venv\Scripts\python.exe research_app.py
```

Open http://127.0.0.1:5050. Stop an older app process using that port before launch. `research_app.py` imports the unchanged original application and registers research routes and additive response hooks. The existing upload/prediction controls remain. Neural calibration selection retained identity, so their scores remain explicitly uncalibrated. The original raw decision rule is unchanged. PDF output gains a probability/limitations disclosure page. The original `python app.py` remains available without this extension.

The new routes are `GET /api/research/summary`, `/research-assets/<name>`, and PNG-only `/research-artifacts/<name>`. Case-level error CSVs and clinical metadata are not exposed through artifact routes. Uploads continue to be handled in memory.

## Executed methodology

- Audit all original metadata, deaths/invalid endpoints, repeated patients, case accounting, group splits, training-only target threshold and normalization, every raw operation sequence, frozen-model hashes, and saved predictions.
- Compare all model pairs using 2,000 **paired patient-cluster** bootstrap draws. Store the distributions and marginal 95% percentile difference intervals. No confirmatory superiority claim or multiplicity adjustment.
- Select identity, Platt, or isotonic calibration using five-fold patient-group cross-fitted **validation** Brier score. Refit only on validation. Test intercept/slope regressions are diagnostics, never deployed calibration transforms. ECE uses ten equal-width bins.
- Evaluate age, sex, surgery category/duration, ASA, and emergency/elective subgroups. Withhold unstable metrics; use patient bootstrap intervals only where supported.
- Retrain nine Transformer variants: six full-window feature sets and three all-feature 30/60/90-minute prefixes. Refit each prefix's normalization using only its training prefix. Preserve original patient splits and labels. No model is promoted. This is not a full factorial design or an all-model-family ablation.
- Describe FP/FN physiology, missingness, variability, procedure and stay-range associations without causal interpretation.
- Validate original 32-step logit Integrated Gradients against numerical completeness at 128/256 steps, normalization/feature/mask alignment, padding, and small input perturbations. Fixed-missingness baseline deliberately excludes missingness contribution.

All experiments are secondary analyses of an already examined single-source holdout. Bootstrap intervals do not capture retraining uncertainty. Early-prefix models include shorter completed operations and are not a strict ongoing-surgery landmark cohort. Recording length can remain predictive even in one-signal models.

## Reproduce as a new version

Do not overwrite these completed research artifacts. Select a **new** `RUN` in `research/common.py` before a new complete research run; the source experiment stays frozen. The following are the execution stages:

```powershell
.\.venv\Scripts\python.exe research/audit.py
# If CPU verification differs, use the preserved audit and original CUDA runtime:
.\.venv\Scripts\python.exe scripts/run_cuda.py --torch-site PATH_TO_CUDA_TORCH_SITE_PACKAGES research/verify_cuda_predictions.py
.\.venv\Scripts\python.exe research/statistics.py
.\.venv\Scripts\python.exe research/calibration.py
.\.venv\Scripts\python.exe research/subgroups.py
.\.venv\Scripts\python.exe research/error_analysis.py
.\.venv\Scripts\python.exe scripts/run_cuda.py --torch-site PATH_TO_CUDA_TORCH_SITE_PACKAGES research/ablations.py
.\.venv\Scripts\python.exe scripts/run_cuda.py --torch-site PATH_TO_CUDA_TORCH_SITE_PACKAGES research/explainability.py
.\.venv\Scripts\python.exe research/reporting.py --dashboard
.\.venv\Scripts\python.exe -m pytest -q
```

This computer used the compatible, read-only CUDA torch runtime at `C:\Users\Admin\.vscode\CharacterStudio\.venv\Lib\site-packages`, on the RTX 4070 SUPER. It imports NumPy 1.26, while standalone research uses NumPy 2.2. A NumPy-version compatibility issue in bootstrap integration was corrected before completing ablations; no completed experiment/checkpoint was overwritten. Repeated executions of completed calibration/audit stages are intentionally refused. Ablations preserve completed per-variant results on resume.

Final report: `results/RESEARCH_VALIDATION_REPORT.md`. Evidence: `results/statistical_validation/advanced_20261110/`. Separate calibration parameters: `models/calibration/advanced_20261110/`. Separate ablation checkpoints: `models/checkpoints/advanced_20261110/ablations/`. Protected original-file SHA-256 values are retained in `preserved_manifest.json`; final tests verify they remain unchanged.

Not clinically validated. Not for clinical deployment.
