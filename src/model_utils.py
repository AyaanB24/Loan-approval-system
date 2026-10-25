"""Training and prediction utilities for the educational loan-approval demo."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "loan_approval_demo.csv"
MODEL_PATH = ROOT / "models" / "loan_model.joblib"
TARGET = "Loan_Status"
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
CATEGORICAL = ["Property_Area", "Education", "Self_Employed"]
FEATURES = NUMERIC + CATEGORICAL

def build_pipeline(model_name="Random Forest"):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    estimator = (RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42)
                 if model_name == "Random Forest" else LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    return Pipeline([("preprocess", prep), ("model", estimator)])

def train_model(model_name="Random Forest", data=None):
    df = pd.read_csv(DATA_PATH) if data is None else data.copy()
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    df = df[FEATURES + [TARGET]].copy().dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title()
    df = df[df[TARGET].isin(["Approved", "Rejected"])]
    if len(df) < 30 or df[TARGET].nunique() < 2:
        raise ValueError("Need at least 30 usable rows and both Approved and Rejected labels.")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    pipe = build_pipeline(model_name)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    classes = list(pipe.named_steps["model"].classes_)
    approved_idx = classes.index("Approved") if "Approved" in classes else None
    metrics = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, pos_label="Approved", zero_division=0), "recall": recall_score(y_test, pred, pos_label="Approved", zero_division=0), "f1": f1_score(y_test, pred, pos_label="Approved", zero_division=0), "confusion_matrix": confusion_matrix(y_test, pred, labels=["Rejected", "Approved"]).tolist(), "test_rows": len(y_test), "train_rows": len(y_train), "class_labels": classes, "report": classification_report(y_test, pred, zero_division=0, output_dict=True)}
    if approved_idx is not None and len(set(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score((y_test == "Approved").astype(int), proba[:, approved_idx])
    joblib.dump({"pipeline": pipe, "model_name": model_name, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    return pipe, metrics

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)
