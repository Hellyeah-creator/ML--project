
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, accuracy_score,
                              classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_PATH = "/home/claude/churn_project/data/customer_churn.csv"
PLOTS_DIR = "/home/claude/churn_project/plots"
OUT_DIR = "/home/claude/churn_project/outputs"

# ---------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Overall churn rate
churn_counts = df["Churn"].value_counts()
axes[0].pie(churn_counts, labels=["No Churn", "Churn"], autopct="%1.1f%%",
            colors=["#4C72B0", "#DD8452"], startangle=90)
axes[0].set_title("Overall Churn Rate")

# Churn by contract type
contract_churn = df.groupby("Contract")["Churn"].mean().sort_values() * 100
sns.barplot(x=contract_churn.values, y=contract_churn.index, ax=axes[1],
            palette="viridis")
axes[1].set_xlabel("Churn Rate (%)")
axes[1].set_title("Churn Rate by Contract Type")

# Tenure distribution by churn
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack",
             bins=24, ax=axes[2], palette=["#4C72B0", "#DD8452"])
axes[2].set_title("Tenure Distribution by Churn")
axes[2].legend(["Churn", "No Churn"])

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/eda_overview.png", dpi=150)
plt.close()
print("Saved EDA overview plot")

# Correlation heatmap (numeric + encoded key categoricals)
eda_df = df.copy()
eda_df["Contract_M2M"] = (eda_df["Contract"] == "Month-to-month").astype(int)
eda_df["Fiber"] = (eda_df["InternetService"] == "Fiber optic").astype(int)
eda_df["ElecCheck"] = (eda_df["PaymentMethod"] == "Electronic check").astype(int)
eda_df["TechSupport_Yes"] = (eda_df["TechSupport"] == "Yes").astype(int)
corr_cols = ["Churn", "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
             "Contract_M2M", "Fiber", "ElecCheck", "TechSupport_Yes"]
plt.figure(figsize=(7, 6))
sns.heatmap(eda_df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation with Churn")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved correlation heatmap")

# ---------------------------------------------------------------
# 3. PREPROCESS
# ---------------------------------------------------------------
model_df = df.drop(columns=["customerID"])

binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for c in binary_cols:
    model_df[c] = model_df[c].map({"Yes": 1, "No": 0})
model_df["gender"] = model_df["gender"].map({"Male": 1, "Female": 0})

multi_cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                   "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                   "Contract", "PaymentMethod"]
model_df = pd.get_dummies(model_df, columns=multi_cat_cols, drop_first=True)

X = model_df.drop(columns=["Churn"])
y = model_df["Churn"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------
# 4. TRAIN & COMPARE MODELS
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42),
}

sw_train = np.where(y_train == 1, (y_train == 0).sum() / (y_train == 1).sum(), 1.0)

results = {}
fitted = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    elif name == "Gradient Boosting":
        model.fit(X_train, y_train, sample_weight=sw_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted[name] = {"model": model, "preds": preds, "proba": proba}
    print(f"\n{name}")
    for k, v in results[name].items():
        print(f"  {k}: {v:.3f}")

results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv")
print("\n=== Model comparison ===")
print(results_df.round(3))

best_name = results_df.index[0]
best = fitted[best_name]
print(f"\nBest model by ROC-AUC: {best_name}")

# ---------------------------------------------------------------
# 5. EVALUATION PLOTS
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Model comparison bar chart
results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
    kind="bar", ax=axes[0], colormap="viridis")
axes[0].set_title("Model Comparison")
axes[0].set_ylim(0, 1)
axes[0].legend(loc="lower right", fontsize=8)
axes[0].tick_params(axis="x", rotation=20)

# Confusion matrix for best model
cm = confusion_matrix(y_test, best["preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"],
            ax=axes[1])
axes[1].set_title(f"Confusion Matrix – {best_name}")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# ROC curves for all models
for name in models:
    RocCurveDisplay.from_predictions(
        y_test, fitted[name]["proba"], name=name, ax=axes[2]
    )
axes[2].set_title("ROC Curves")
axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/model_evaluation.png", dpi=150)
plt.close()
print("Saved model evaluation plot")

# Feature importance (from Random Forest, or best tree model)
tree_model_name = "Random Forest" if "Random Forest" in fitted else best_name
importances = fitted[tree_model_name]["model"].feature_importances_
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(12)

plt.figure(figsize=(8, 6))
sns.barplot(x=imp_series.values, y=imp_series.index, palette="mako")
plt.title(f"Top 12 Feature Importances ({tree_model_name})")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150)
plt.close()
print("Saved feature importance plot")

# ---------------------------------------------------------------
# 6. SAVE MODEL + REPORT
# ---------------------------------------------------------------
joblib.dump({"model": best["model"], "scaler": scaler if best_name == "Logistic Regression" else None,
             "features": feature_names, "model_name": best_name},
            f"{OUT_DIR}/best_churn_model.joblib")

report = classification_report(y_test, best["preds"], target_names=["No Churn", "Churn"])
with open(f"{OUT_DIR}/classification_report.txt", "w") as f:
    f.write(f"Best model: {best_name}\n\n{report}")

summary = {
    "best_model": best_name,
    "test_set_size": len(y_test),
    "churn_rate_in_data": float(y.mean()),
    "metrics": results,
    "top_features": imp_series.to_dict(),
}
with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone. Outputs saved to", OUT_DIR, "and", PLOTS_DIR)
