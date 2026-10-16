"""
Automated unit and integration tests for Heart Disease Prediction Pipeline.
"""

import os
import joblib
import pandas as pd
import numpy as np
from src.data_loader import load_data
from src.train import build_preprocessor

def test_data_loading():
    df = load_data()
    assert not df.empty
    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})
    assert df.shape[1] == 14

def test_preprocessor_transformation():
    df = pd.read_csv("data/heart_disease_cleaned.csv")
    X = df.drop(columns=["target"])
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    assert X_trans.shape[0] == len(df)
    assert not np.isnan(X_trans).any()

def test_saved_model_inference():
    model_path = "models/heart_disease_model.joblib"
    assert os.path.exists(model_path)
    model = joblib.load(model_path)
    
    sample_patient = pd.DataFrame([{
        "age": 55, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 140, "exang": 0,
        "oldpeak": 1.2, "slope": 2, "ca": 1.0, "thal": 6.0
    }])
    
    pred = model.predict(sample_patient)
    proba = model.predict_proba(sample_patient)
    
    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)
    assert np.isclose(np.sum(proba), 1.0)
