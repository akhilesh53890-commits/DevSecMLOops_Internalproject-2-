"""
End-to-End Model Training, CV Comparison, Optuna Hyperparameter Tuning,
Evaluation, Feature Importance, and Model Serialization.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
import optuna

# Filter minor convergence or formatting warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seed
SEED = 42
np.random.seed(SEED)

DATA_PATH = "data/heart_disease_cleaned.csv"
REPORTS_DIR = "reports"
MODELS_DIR = "models"
MODEL_FILE = os.path.join(MODELS_DIR, "heart_disease_model.joblib")
METRICS_FILE = os.path.join(REPORTS_DIR, "metrics.json")

# Define feature groups
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

def build_preprocessor():
    """Builds the scikit-learn ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor

def evaluate_cv_models(X_train, y_train, cv):
    """Compares Logistic Regression, Random Forest, and XGBoost using Stratified 5-Fold CV."""
    print("\n" + "="*60)
    print("STEP 1: 5-Fold Stratified Cross-Validation Comparison")
    print("="*60)

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1
        )
    }

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_summary = {}

    for name, clf in candidate_models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
        cv_summary[name] = {
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std": float(np.std(scores["test_accuracy"])),
            "precision_mean": float(np.mean(scores["test_precision"])),
            "precision_std": float(np.std(scores["test_precision"])),
            "recall_mean": float(np.mean(scores["test_recall"])),
            "recall_std": float(np.std(scores["test_recall"])),
            "f1_mean": float(np.mean(scores["test_f1"])),
            "f1_std": float(np.std(scores["test_f1"])),
            "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        }
        print(f"\nModel: {name}")
        print(f"  ROC-AUC:  {cv_summary[name]['roc_auc_mean']:.4f} (+/- {cv_summary[name]['roc_auc_std']:.4f})")
        print(f"  F1-Score: {cv_summary[name]['f1_mean']:.4f} (+/- {cv_summary[name]['f1_std']:.4f})")
        print(f"  Accuracy: {cv_summary[name]['accuracy_mean']:.4f} (+/- {cv_summary[name]['accuracy_std']:.4f})")

    best_model_name = max(cv_summary.keys(), key=lambda m: (cv_summary[m]["roc_auc_mean"], cv_summary[m]["f1_mean"]))
    print(f"\n>>> Best Baseline Model selected for Optuna tuning: {best_model_name} (ROC-AUC: {cv_summary[best_model_name]['roc_auc_mean']:.4f})")
    return cv_summary, best_model_name

def tune_with_optuna(best_model_name, X_train, y_train, cv, n_trials=40):
    """Tunes hyperparameters of the chosen best model with Optuna."""
    print("\n" + "="*60)
    print(f"STEP 2: Optuna Hyperparameter Tuning for {best_model_name} ({n_trials} trials)")
    print("="*60)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if best_model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 3.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "eval_metric": "logloss",
                "random_state": SEED,
                "n_jobs": 1
            }
            clf = XGBClassifier(**params)
        elif best_model_name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 350, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": SEED,
                "n_jobs": 1
            }
            clf = RandomForestClassifier(**params)
        else: # Logistic Regression
            l1_ratio = trial.suggest_categorical("l1_ratio", [0.0, 0.5, 1.0])
            params = {
                "C": trial.suggest_float("C", 1e-3, 50.0, log=True),
                "l1_ratio": l1_ratio,
                "solver": "saga",
                "max_iter": 2000,
                "random_state": SEED
            }
            clf = LogisticRegression(**params)

        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring="roc_auc", return_train_score=False)
        return float(np.mean(scores["test_score"]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Optuna Optimization Complete!")
    print(f"Best Trial ROC-AUC: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params, study.best_value

def build_final_estimator(best_model_name, best_params):
    """Instantiates the tuned classifier with optimal hyperparameters."""
    if best_model_name == "XGBoost":
        return XGBClassifier(
            **best_params,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1
        )
    elif best_model_name == "Random Forest":
        return RandomForestClassifier(
            **best_params,
            random_state=SEED,
            n_jobs=1
        )
    else:
        return LogisticRegression(
            **best_params,
            solver="saga",
            max_iter=2000,
            random_state=SEED
        )

def train_and_evaluate():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Stratified Train-Test Split (80% Train, 20% Held-Out Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"Data Split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Step 1: Baseline comparison across 3 algorithms
    cv_summary, best_model_name = evaluate_cv_models(X_train, y_train, cv)

    # Step 2: Optuna Tuning
    best_params, optuna_best_auc = tune_with_optuna(best_model_name, X_train, y_train, cv, n_trials=40)

    # Step 3: Train final tuned model on entire training set
    print("\n" + "="*60)
    print("STEP 3: Fitting Tuned Pipeline on Full Training Set")
    print("="*60)

    final_clf = build_final_estimator(best_model_name, best_params)
    final_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", final_clf)
    ])
    final_pipeline.fit(X_train, y_train)

    # Step 4: Evaluate on held-out test set
    print("\n" + "="*60)
    print("STEP 4: Evaluating on Held-Out Test Set")
    print("="*60)

    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")
    print(f"Test ROC-AUC:   {roc_auc:.4f}")

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_array = np.array(cm)
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Disease (0)", "Disease (1)"],
        yticklabels=["No Disease (0)", "Disease (1)"]
    )
    plt.title(f"Confusion Matrix: {best_model_name}\n(Held-out Test Set, N={len(y_test)})", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="semibold")
    plt.ylabel("True Class", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # Plot 2: ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"{best_model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="semibold")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()
    roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve plot to {roc_path}")

    # Plot 3: Feature Importance
    plt.figure(figsize=(10, 7))
    preprocessor = final_pipeline.named_steps["preprocessor"]
    fitted_clf = final_pipeline.named_steps["classifier"]
    
    # Get encoded feature names
    feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    if hasattr(fitted_clf, "feature_importances_"):
        importances = fitted_clf.feature_importances_
        imp_title = f"Gini / Tree Feature Importance ({best_model_name})"
    elif hasattr(fitted_clf, "coef_"):
        importances = np.abs(fitted_clf.coef_[0])
        imp_title = f"Absolute Feature Coefficients ({best_model_name})"
    else:
        importances = np.ones(len(clean_feature_names))
        imp_title = "Feature Weights"

    feat_df = pd.DataFrame({
        "Feature": clean_feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    # Top features
    top_feat_df = feat_df.tail(15)
    plt.barh(top_feat_df["Feature"], top_feat_df["Importance"], color="#2b5c8f", edgecolor="#1a3b5c", alpha=0.85)
    plt.title(imp_title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Relative Importance Score", fontsize=11, fontweight="semibold")
    plt.ylabel("Clinical Predictor", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    fi_path = os.path.join(REPORTS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to {fi_path}")

    # Step 5: Save model to models/
    joblib.dump(final_pipeline, MODEL_FILE)
    print(f"Saved complete inference pipeline to {MODEL_FILE}")

    # Step 6: Save metrics.json
    metrics_data = {
        "project": "Heart Disease Risk Prediction",
        "dataset": "UCI Cleveland Heart Disease",
        "best_model": best_model_name,
        "cv_comparison": cv_summary,
        "optuna_best_params": best_params,
        "optuna_best_cv_auc": optuna_best_auc,
        "test_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": clf_report
        }
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved comprehensive evaluation metrics to {METRICS_FILE}")

    return metrics_data

if __name__ == "__main__":
    train_and_evaluate()
