# RecoveryLens AI
### Postoperative Hospital Stay Prediction Using Deep Learning

RecoveryLens AI is a healthcare AI project that analyzes a patient's physiological measurements recorded during surgery and predicts whether they are likely to have a prolonged hospital stay after surgery.

It uses a **Temporal Transformer** trained on real surgical data from the **VitalDB dataset**.

The application also provides interactive physiological graphs, AI explanations, and downloadable prediction reports.

**Tech Stack:** Python · PyTorch · Flask · HTML · CSS · JavaScript · Chart.js

---

## Model Performance

Evaluated on **1,186 held-out surgical cases**.

| Metric | Result |
|---|---:|
| **Accuracy** | **82.72%** |
| **AUROC** | **86.39%** |
| Precision | 64.33% |
| Recall | 66.32% |
| F1-score | 65.31% |

The model correctly classified **981 out of 1,186 cases**.

These results are from retrospective testing, not clinical validation.

## How It Works

```text
       Upload Physiological CSV
                  |
                  v
           Data Preprocessing
                  |
                  v
          Temporal Transformer
                  |
                  v
       Hospital Stay Prediction
                  |
                  v
         Results & AI Explanation
                  |
                  v
            Download Report
```

**1. Upload:** Upload physiological measurements recorded during surgery.

**2. Preprocessing:** The system prepares the time-series data and handles missing measurements.

**3. Prediction:** The Temporal Transformer analyzes the measurements and predicts prolonged hospitalization.

**4. Explainability:** Interactive charts and Integrated Gradients help visualize the model's behavior.

**5. Report:** Download the prediction results as a PDF.

## Dataset

**Dataset:** [VitalDB](https://vitaldb.net/dataset/)

The project uses seven physiological signals:

| Signal | Description |
|---|---|
| HR | Heart rate |
| SpO₂ | Oxygen saturation |
| SBP | Systolic blood pressure |
| DBP | Diastolic blood pressure |
| MAP | Mean arterial pressure |
| EtCO₂ | End-tidal carbon dioxide |
| RR | Respiratory rate |

### Dataset Split

| Split | Cases |
|---|---:|
| Training | 3,981 |
| Validation | 952 |
| Testing | 1,186 |
| **Total** | **6,119** |

The model predicts whether postoperative hospitalization exceeds **7.39610 days**, the threshold calculated from the training data.

## Model

The project uses a **Temporal Transformer**, a deep learning architecture designed to process sequential data.

Unlike models that rely only on summary statistics, the Transformer learns relationships between physiological measurements across time.

The model uses 14 input features: seven physiological signals and seven indicators of missing measurements.

### Model Comparison

| Model | AUROC |
|---|---:|
| **Temporal Transformer** | **0.8639** |
| GRU | 0.8580 |
| XGBoost | 0.7983 |

The Transformer achieved the highest AUROC, although its advantage over GRU was not statistically established.

## Application Features

- **Hospital stay prediction:** Displays the model's predicted probability and classification.
- **Physiological monitoring:** Interactive graphs for surgical measurements.
- **Explainable AI:** Integrated Gradients highlights influential signals and time periods.
- **Missing-data handling:** Supports incomplete physiological recordings.
- **PDF reports:** Download the prediction summary and visualizations.
- **Responsive dashboard:** Works on desktop and mobile screens.

## Installation

**1. Clone the repository**

```bash
git clone <YOUR_REPOSITORY_URL>
cd RecoveryLens_AI
```

**2. Create a virtual environment**

```powershell
python -m venv .venv
```

**3. Install dependencies**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**4. Run the application**

```powershell
.\.venv\Scripts\python.exe transformer_app.py
```

Open **http://127.0.0.1:5050** in your browser.

## Input CSV Format

```csv
time_min,HR,SpO2,SBP,DBP,MAP,EtCO2,RR
0,82,99,120,80,93,35,14
1,85,98,118,78,91,36,14
2,91,98,110,74,86,35,15
3,97,97,102,68,79,34,15
4,105,96,95,62,73,33,16
```

*The example values above are synthetic and only demonstrate the CSV format.*

The application requires sufficient physiological data and supports missing observations.

The public repository does not include patient-level recordings. You can upload your own compatible CSV file.

## Real Patient Testing

The application was tested with a real held-out VitalDB recording.

| Metric | Result |
|---|---|
| Actual postoperative stay | 4.39 days |
| Model prediction | 12.89% |
| Actual classification | Not prolonged |
| Predicted classification | Not prolonged |
| **Prediction correct** | **Yes** |

CSV upload, inference, physiological graphs, AI explanations, and PDF generation were successfully tested.

## Limitations

RecoveryLens AI is a **research prototype**, not a clinically approved medical tool.

Its predictions are not clinically calibrated, and the model can miss prolonged hospitalizations. External hospital validation is required before considering clinical use.

## Acknowledgment

Dataset: [VitalDB](https://vitaldb.net/dataset/)

Lee HC et al. *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients.* Scientific Data (2022).

---

**RecoveryLens AI — Using physiological time-series data to explore postoperative recovery prediction.**

*For research and educational purposes only. Not for clinical decision-making.*
