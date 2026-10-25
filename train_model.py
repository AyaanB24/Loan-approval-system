from src.model_utils import train_model

if __name__ == "__main__":
    for name in ["Logistic Regression", "Random Forest"]:
        _, metrics = train_model(name)
        print(f"\n{name}")
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "train_rows", "test_rows"]:
            if key in metrics:
                print(f"{key}: {metrics[key]:.3f}" if isinstance(metrics[key], float) else f"{key}: {metrics[key]}")
    print("\nSaved model: models/loan_model.joblib (Random Forest is the final model).")
