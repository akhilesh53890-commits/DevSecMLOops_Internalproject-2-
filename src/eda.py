"""
Exploratory Data Analysis (EDA) module for Heart Disease Risk Prediction.
Generates informative, publication-quality visualizations saved to reports/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans, sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

REPORTS_DIR = "reports"
DATA_PATH = "data/heart_disease_cleaned.csv"

def generate_eda_reports():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        from src.data_loader import load_data
        df = load_data()
    else:
        df = pd.read_csv(DATA_PATH)

    print(f"Loaded dataset for EDA: {df.shape}")

    # Set aesthetic palette
    palette = ["#2b5c8f", "#d95f02"] # Blue (No Disease), Orange/Red (Disease)

    # 1. Target Class Balance Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = df["target"].value_counts()
    labels = ["No Disease (0)", "Heart Disease (1)"]
    colors = palette

    # Bar chart
    sns.barplot(x=labels, y=counts.values, ax=axes[0], palette=colors, hue=labels, legend=False)
    axes[0].set_title("Target Class Distribution (Counts)", fontsize=14, fontweight="bold", pad=12)
    axes[0].set_ylabel("Patient Count", fontsize=12)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 3, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontsize=11, fontweight="semibold")
    axes[0].set_ylim(0, max(counts.values) * 1.15)

    # Donut chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2),
        textprops=dict(fontsize=11)
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")
    axes[1].set_title("Target Proportion Ratio", fontsize=14, fontweight="bold", pad=12)

    plt.suptitle("Heart Disease Dataset: Class Balance", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    class_balance_path = os.path.join(REPORTS_DIR, "class_balance.png")
    plt.savefig(class_balance_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved class balance plot to {class_balance_path}")

    # 2. Correlation Matrix Heatmap
    plt.figure(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.6,
        vmax=0.6,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
    )
    plt.title("Correlation Matrix of Clinical Attributes", fontsize=15, fontweight="bold", pad=14)
    plt.tight_layout()
    corr_path = os.path.join(REPORTS_DIR, "correlation_matrix.png")
    plt.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation matrix to {corr_path}")

    # 3. Continuous Feature Distributions by Target Class
    continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    target_labels = {0: "No Disease", 1: "Disease"}
    df_plot = df.copy()
    df_plot["Condition"] = df_plot["target"].map(target_labels)

    for idx, col in enumerate(continuous_features):
        ax = axes[idx]
        sns.histplot(
            data=df_plot,
            x=col,
            hue="Condition",
            kde=True,
            palette={"No Disease": palette[0], "Disease": palette[1]},
            bins=18,
            ax=ax,
            alpha=0.45,
            edgecolor=None
        )
        ax.set_title(f"Distribution of {col.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("Density / Count", fontsize=10)

    # In the 6th subplot, show Resting BP vs Max Heart Rate scatter with target hue
    ax_scatter = axes[5]
    sns.scatterplot(
        data=df_plot,
        x="thalach",
        y="trestbps",
        hue="Condition",
        palette={"No Disease": palette[0], "Disease": palette[1]},
        alpha=0.75,
        s=45,
        ax=ax_scatter
    )
    ax_scatter.set_title("Resting BP vs Max Heart Rate (thalach)", fontsize=12, fontweight="bold")
    ax_scatter.set_xlabel("Max Heart Rate (thalach)", fontsize=11)
    ax_scatter.set_ylabel("Resting Blood Pressure (trestbps)", fontsize=11)

    plt.suptitle("Clinical Feature Distributions Stratified by Heart Disease Status", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    dist_path = os.path.join(REPORTS_DIR, "feature_distributions.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature distributions plot to {dist_path}")

    # 4. Key Categorical Risk Predictors
    categorical_features = ["sex", "cp", "exang", "slope"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, col in enumerate(categorical_features):
        ax = axes[idx]
        ct = pd.crosstab(df[col], df["target"], normalize="index") * 100
        ct.columns = ["No Disease (%)", "Disease (%)"]
        ct.plot(kind="bar", stacked=True, color=palette, ax=ax, alpha=0.85, edgecolor="white")
        ax.set_title(f"Prevalence by {col.upper()}", fontsize=13, fontweight="bold")
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(loc="upper right")
        for p in ax.patches:
            height = p.get_height()
            if height > 5:
                ax.annotate(f"{height:.0f}%",
                            (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.),
                            ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    plt.suptitle("Risk Breakdown Across Key Categorical Predictors", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    cat_path = os.path.join(REPORTS_DIR, "categorical_features.png")
    plt.savefig(cat_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved categorical features plot to {cat_path}")

if __name__ == "__main__":
    generate_eda_reports()
    print("EDA Visualizations successfully generated.")
