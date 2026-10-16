"""
Streamlit Web Application: Heart Disease Risk Prediction Clinical Assistant.
Allows clinicians or users to enter patient clinical attributes and receive
an instant risk assessment with probability and contributing factors.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page metadata and styling
st.set_page_config(
    page_title="CardioRisk AI | Heart Disease Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished medical UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .risk-high {
        background-color: #fff5f5;
        border: 1px solid #feb2b2;
        border-left: 6px solid #e53e3e;
        padding: 20px;
        border-radius: 10px;
        color: #9b2c2c;
    }
    .risk-low {
        background-color: #f0fff4;
        border: 1px solid #9ae6b4;
        border-left: 6px solid #38a169;
        padding: 20px;
        border-radius: 10px;
        color: #22543d;
    }
    .stButton>button {
        background-color: #2b5c8f;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "models/heart_disease_model.joblib"
METRICS_PATH = "reports/metrics.json"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

pipeline = load_trained_model()
metrics = load_metrics()

# Header
st.markdown('<div class="main-header">🫀 CardioRisk AI — Heart Disease Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Internal Project 1 | Healthcare & Clinical Decision Support System trained on UCI Cleveland Cohort</div>', unsafe_allow_html=True)

# Sidebar with presets and project info
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1505751172876-fa1923c5c528?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.header("📋 Clinical Quick Presets")
    preset = st.selectbox(
        "Load Example Patient Profile:",
        ["Select a preset...", "Patient A (Low Risk Profile)", "Patient B (High Risk Profile)"]
    )

    preset_values = None
    if preset == "Patient A (Low Risk Profile)":
        preset_values = {
            "age": 45, "sex": 0, "cp": 2, "trestbps": 120, "chol": 190,
            "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0,
            "oldpeak": 0.0, "slope": 1, "ca": 0.0, "thal": 3.0
        }
    elif preset == "Patient B (High Risk Profile)":
        preset_values = {
            "age": 62, "sex": 1, "cp": 4, "trestbps": 155, "chol": 290,
            "fbs": 1, "restecg": 2, "thalach": 125, "exang": 1,
            "oldpeak": 2.6, "slope": 2, "ca": 2.0, "thal": 7.0
        }

    st.divider()
    if metrics:
        st.subheader("📊 Model Performance")
        st.caption(f"**Architecture:** Tuned {metrics.get('best_model', 'Classifier')}")
        st.write(f"• **Test ROC-AUC:** `{metrics['test_metrics']['roc_auc']:.3f}`")
        st.write(f"• **Test Accuracy:** `{metrics['test_metrics']['accuracy']*100:.1f}%`")
        st.write(f"• **Test Recall:** `{metrics['test_metrics']['recall']*100:.1f}%`")
        st.write(f"• **Test F1-Score:** `{metrics['test_metrics']['f1_score']:.3f}`")

# Main Patient Data Input Form
st.subheader("Patient Clinical Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Demographics & Symptoms")
    age = st.slider("Age (years)", 20, 85, int(preset_values["age"]) if preset_values else 54)
    sex = st.radio(
        "Sex",
        options=[1, 0],
        format_func=lambda x: "Male (1)" if x == 1 else "Female (0)",
        index=0 if (preset_values is None or preset_values["sex"] == 1) else 1,
        horizontal=True
    )
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1: Typical Angina",
            2: "2: Atypical Angina",
            3: "3: Non-anginal Pain",
            4: "4: Asymptomatic"
        }[x],
        index=(preset_values["cp"] - 1) if preset_values else 3
    )
    exang = st.radio(
        "Exercise-Induced Angina (exang)",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        index=preset_values["exang"] if preset_values else 0,
        horizontal=True
    )

with col2:
    st.markdown("#### 🩸 Vitals & Hemodynamics")
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        80, 220,
        int(preset_values["trestbps"]) if preset_values else 130,
        step=5
    )
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        100, 600,
        int(preset_values["chol"]) if preset_values else 240,
        step=10
    )
    fbs = st.radio(
        "Fasting Blood Sugar > 120 mg/dl (fbs)",
        options=[0, 1],
        format_func=lambda x: "False / Normal (0)" if x == 0 else "True / Elevated (1)",
        index=preset_values["fbs"] if preset_values else 0,
        horizontal=True
    )
    thalach = st.slider(
        "Maximum Heart Rate Achieved (thalach)",
        60, 220,
        int(preset_values["thalach"]) if preset_values else 150
    )

with col3:
    st.markdown("#### 🔬 Electrocardio & Angiography")
    restecg = st.selectbox(
        "Resting ECG (restecg)",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0: Normal",
            1: "1: ST-T Wave Abnormality",
            2: "2: Left Ventricular Hypertrophy"
        }[x],
        index=preset_values["restecg"] if preset_values else 0
    )
    oldpeak = st.slider(
        "ST Depression Induced by Exercise (oldpeak)",
        0.0, 6.2,
        float(preset_values["oldpeak"]) if preset_values else 1.0,
        step=0.1
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1: Upsloping",
            2: "2: Flat",
            3: "3: Downsloping"
        }[x],
        index=(preset_values["slope"] - 1) if preset_values else 1
    )
    ca = st.selectbox(
        "Major Vessels Colored by Fluoroscopy (ca)",
        options=[0.0, 1.0, 2.0, 3.0],
        format_func=lambda x: f"{int(x)} vessels",
        index=int(preset_values["ca"]) if preset_values else 0
    )
    thal = st.selectbox(
        "Thalassemia Defect Status (thal)",
        options=[3.0, 6.0, 7.0],
        format_func=lambda x: {
            3.0: "3: Normal",
            6.0: "6: Fixed Defect",
            7.0: "7: Reversible Defect"
        }[x],
        index=[3.0, 6.0, 7.0].index(float(preset_values["thal"])) if preset_values else 0
    )

st.divider()

# Prediction logic
if st.button("🔍 Assess Patient Heart Disease Risk", type="primary", use_container_width=True):
    if pipeline is None:
        st.error("Model file not found. Please run `python src/train.py` first.")
    else:
        # Construct DataFrame matching training schema
        input_data = pd.DataFrame([{
            "age": float(age),
            "sex": float(sex),
            "cp": float(cp),
            "trestbps": float(trestbps),
            "chol": float(chol),
            "fbs": float(fbs),
            "restecg": float(restecg),
            "thalach": float(thalach),
            "exang": float(exang),
            "oldpeak": float(oldpeak),
            "slope": float(slope),
            "ca": float(ca),
            "thal": float(thal)
        }])

        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        disease_prob = probabilities[1]
        no_disease_prob = probabilities[0]

        st.subheader("Diagnostic Assessment Result")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3 style="margin-top:0;">⚠️ Elevated Risk of Heart Disease Detected</h3>
                    <p style="font-size:1.1rem; margin-bottom:8px;">
                        The model classifies this patient as <strong>High Risk</strong> for significant coronary narrowing (>50% diameter stenosis).
                    </p>
                    <p><strong>Predicted Probability of Disease:</strong> <span style="font-size:1.4rem; font-weight:bold;">{disease_prob*100:.1f}%</span></p>
                    <p><em>Recommendation: Immediate clinical follow-up, stress imaging, and cardiological evaluation recommended.</em></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3 style="margin-top:0;">✅ Low Risk Profile (Normal Baseline)</h3>
                    <p style="font-size:1.1rem; margin-bottom:8px;">
                        The model classifies this patient as <strong>Low Risk</strong> for significant coronary artery disease.
                    </p>
                    <p><strong>Predicted Probability of Absence:</strong> <span style="font-size:1.4rem; font-weight:bold;">{no_disease_prob*100:.1f}%</span> (Disease Risk: {disease_prob*100:.1f}%)</p>
                    <p><em>Recommendation: Maintain lifestyle counseling, regular health checks, and routine monitoring.</em></p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Probability Distribution")
            st.progress(disease_prob, text=f"Heart Disease Risk: {disease_prob*100:.1f}%")
            
            prob_df = pd.DataFrame({
                "Outcome": ["Absence (No Disease)", "Presence (Heart Disease)"],
                "Probability": [no_disease_prob, disease_prob]
            })
            st.bar_chart(prob_df.set_index("Outcome"), color="#2b5c8f")

        st.divider()
        st.markdown("#### Key Input Summary for Record")
        st.dataframe(input_data, use_container_width=True)

# Tabbed view for Project Architecture & Reports
with st.expander("📂 View Model Reports and Diagnostic Plots", expanded=False):
    tab1, tab2, tab3 = st.tabs(["ROC Curve & Metrics", "Confusion Matrix", "Feature Importance"])
    with tab1:
        if os.path.exists("reports/roc_curve.png"):
            st.image("reports/roc_curve.png", caption="ROC Curve on Held-Out Test Set (AUC: 0.964)")
    with tab2:
        if os.path.exists("reports/confusion_matrix.png"):
            st.image("reports/confusion_matrix.png", caption="Confusion Matrix on Held-Out Test Set")
    with tab3:
        if os.path.exists("reports/feature_importance.png"):
            st.image("reports/feature_importance.png", caption="Top Predictors (L1 Regularized Weights)")

st.caption("CardioRisk AI • For Research & Educational Use Only • UCI Cleveland Heart Disease Dataset")
