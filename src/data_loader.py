"""
Data ingestion and cleaning module for the UCI Heart Disease dataset.
"""

import os
import urllib.request
import pandas as pd
import numpy as np

UCI_CLEVELAND_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

def load_data(data_dir: str = "data") -> pd.DataFrame:
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "raw_cleveland.data")
    clean_csv_path = os.path.join(data_dir, "heart_disease_cleaned.csv")

    df = None

    # Step 1: Try UCI Archive Cleveland URL
    print(f"Attempting to download Cleveland dataset from {UCI_CLEVELAND_URL}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(UCI_CLEVELAND_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response, open(raw_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully downloaded raw dataset to {raw_path}")

        df = pd.read_csv(raw_path, header=None, names=COLUMN_NAMES, na_values="?")
    except Exception as e:
        print(f"UCI direct download failed ({e}). Attempting OpenML fallback...")
        try:
            from sklearn.datasets import fetch_openml
            print("Fetching heart dataset from OpenML...")
            openml_data = fetch_openml(data_id=43609, as_frame=True, parser="auto")
            df = openml_data.frame
            print("Successfully fetched from OpenML data_id 43609")
        except Exception as e2:
            print(f"OpenML fallback failed ({e2}). Attempting secondary OpenML mirror...")
            from sklearn.datasets import fetch_openml
            openml_data = fetch_openml("heart-disease", version=1, as_frame=True, parser="auto")
            df = openml_data.frame

    if df is None or df.empty:
        raise RuntimeError("Failed to load Heart Disease dataset from any source.")

    # Normalize column names to standard lower case
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map target / num column to 'target'
    if "num" in df.columns and "target" not in df.columns:
        df["target"] = df["num"]
    elif "condition" in df.columns and "target" not in df.columns:
        df["target"] = df["condition"]

    # Binarize target: 0 = no disease, 1 = disease (in Cleveland: 0 vs 1,2,3,4)
    if df["target"].dtype == object or df["target"].dtype.name == "category":
        # Handle string labels if from OpenML (e.g. '0', '1' or 'absence', 'presence')
        target_str = df["target"].astype(str).str.lower()
        df["target"] = target_str.apply(lambda v: 0 if v in ["0", "absence", "no", "negative"] else 1)
    else:
        df["target"] = (df["target"] > 0).astype(int)

    # Convert numeric columns properly, replacing '?' or invalid strings with NaN
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    for col in df.columns:
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save cleaned version
    df.to_csv(clean_csv_path, index=False)
    print(f"Saved processed dataset to {clean_csv_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True).round(4)}")

    return df

if __name__ == "__main__":
    df = load_data()
    print("Data loading completed successfully.")
