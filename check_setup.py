# check_setup.py
import sys


print("MotorGuard — Environment Check")


print(f"\n✅ Python version: {sys.version}")

packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "shap": "shap",
    "mlflow": "mlflow",
    "fastapi": "fastapi",
    "streamlit": "streamlit",
    "evidently": "evidently",
    "imblearn": "imbalanced-learn",
    "joblib": "joblib"
}

all_good = True
for package, label in packages.items():
    try:
        __import__(package)
        print(f"✅ {label} — OK")
    except ImportError:
        print(f"❌ {label} — MISSING")
        all_good = False

print("\n--- Checking dataset access ---")
try:
    import kagglehub
    import pandas as pd
    import os

    path = kagglehub.dataset_download(
        "shivamb/machine-predictive-maintenance-classification"
    )

    # Find the CSV file inside the downloaded folder
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(path, csv_file))

    print(f"✅ Dataset downloaded — {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✅ Columns: {df.columns.tolist()}")
    print(f"✅ File location: {path}")
except Exception as e:
    print(f"❌ Dataset error: {e}")
    all_good = False


if all_good:
    print("🟢 Phase 1 Complete — Environment ready.")
else:
    print("🔴 Some issues found — fix the ❌ items above.")
