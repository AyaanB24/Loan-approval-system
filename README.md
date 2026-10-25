# LoanLens — ML Loan Approval Analysis (Educational Project)

A complete local project with an interactive Streamlit UI, preprocessing pipeline, two ML models, held-out evaluation, demo predictions, dataset explorer, and CSV download.

> **Important:** Included data is synthetic. This is a learning/demo project, not a validated credit underwriting system. Do not use it to approve/reject real loan applications or make real lending decisions.

## Features
- Interactive UI with Overview, Predict, Model Evaluation, Dataset Explorer, and About pages
- Logistic Regression and Random Forest pipelines
- Numeric imputation/scaling and categorical imputation/one-hot encoding
- Stratified 80/20 holdout evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report
- Synthetic sample CSV included; no external data download required
- Download demo data from the UI
- Sensitive personal attributes such as gender and religion are not used as input features

## Requirements
- Python 3.10–3.12 recommended
- Windows, macOS, or Linux

## Run on Windows (PowerShell)
```powershell
cd loan-approval-ml
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Run on macOS/Linux
```bash
cd loan-approval-ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Train/evaluate from terminal
```bash
python train_model.py
```
This trains both model options and writes the final Random Forest artifact to `models/loan_model.joblib`. The UI can also train the selected model.

## Dataset columns
| Column | Meaning |
|---|---|
| ApplicantIncome | Applicant annual income in rupees (synthetic) |
| CoapplicantIncome | Co-applicant annual income in rupees (synthetic) |
| LoanAmount | Requested principal in rupees (synthetic) |
| Loan_Amount_Term | Loan term in months |
| Credit_History | Simplified demo flag (0/1), not a credit score |
| Property_Area | Rural, Semiurban, or Urban |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| Loan_Status | Synthetic target label: Approved / Rejected |

## Project structure
```text
loan-approval-ml/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/loan_approval_demo.csv
├── models/                 # model artifact generated at runtime
└── src/model_utils.py
```

## Important limitations
The target labels are generated from a synthetic probability rule. Therefore, the model learns a simulation rather than real lender behavior. Metrics are for demonstrating a machine-learning workflow only. Real deployment requires representative lawful data, independent validation, fairness and proxy analysis, calibration, explainability, drift monitoring, privacy/security controls, compliance review, and meaningful human oversight.
