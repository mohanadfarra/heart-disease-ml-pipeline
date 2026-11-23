# ===================================================
# Heart Disease Prediction - CS280/CS485 Lab Project
# ===================================================

# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# ============================================================
# 1. LOAD & INSPECT DATA
# ============================================================
data = pd.read_csv("heart.csv")

print("Shape:", data.shape)
print("\nFirst rows:")
data.head()

print("Missing values:\n", data.isnull().sum())
print("\nClass balance:\n", data["target"].value_counts())
print("\nSummary statistics:")
data.describe()

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

data["target"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "tomato"])
axes[0].set_title("Class Distribution (0=No Disease, 1=Disease)")
axes[0].set_xlabel("Target")
axes[0].set_ylabel("Count")
axes[0].tick_params(rotation=0)

data.groupby("target")["age"].plot(kind="kde", ax=axes[1], legend=True)
axes[1].set_title("Age Distribution by Target")
axes[1].set_xlabel("Age")
axes[1].legend(["No Disease", "Disease"])

corr = data.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[2])
axes[2].set_title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()

# ============================================================
# 3. PREPROCESSING
# ============================================================
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ============================================================
# 4. MODELLING + HYPERPARAMETER TUNING
# ============================================================

# --- Logistic Regression ---
lr_params = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, scoring="f1")
lr_grid.fit(X_train_scaled, y_train)
best_lr = lr_grid.best_estimator_
print("Best LR params:", lr_grid.best_params_)

# --- Decision Tree ---
dt_params = {"max_depth": [2, 3, 4, 5], "min_samples_split": [2, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring="f1")
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_
print("Best DT params:", dt_grid.best_params_)

# --- Random Forest ---
rf_params = {"n_estimators": [50, 100], "max_depth": [3, 4, 5], "min_samples_split": [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring="f1")
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)

# ============================================================
# 5. PREDICTIONS
# ============================================================
pred_lr = best_lr.predict(X_test_scaled)
pred_dt = best_dt.predict(X_test)
pred_rf = best_rf.predict(X_test)

model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]
predictions  = [pred_lr, pred_dt, pred_rf]

# ============================================================
# 6. EVALUATION & COMPARISON
# ============================================================

# --- Full classification reports ---
for name, pred in zip(model_names, predictions):
    print("=" * 50)
    print(f"Model: {name}")
    print("=" * 50)
    print(classification_report(y_test, pred, target_names=["No Disease", "Disease"]))

# --- Comparison table ---
def evaluate(name, y_true, y_pred):
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "F1 Score":  round(f1_score(y_true, y_pred), 4),
    }

results = pd.DataFrame([
    evaluate("Logistic Regression", y_test, pred_lr),
    evaluate("Decision Tree",       y_test, pred_dt),
    evaluate("Random Forest",       y_test, pred_rf),
])

print(results.to_string(index=False))

# --- Comparison bar chart ---
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width, results.iloc[0][metrics], width, label="Logistic Regression", color="steelblue")
ax.bar(x,         results.iloc[1][metrics], width, label="Decision Tree",       color="orange")
ax.bar(x + width, results.iloc[2][metrics], width, label="Random Forest",       color="green")

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison - Evaluation Metrics")
ax.legend()
plt.tight_layout()
plt.show()

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, name, pred in zip(axes, model_names, predictions):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_title(f"Confusion Matrix\n{name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
plt.show()

# --- ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 6))

for name, model, X_eval in [
    ("Logistic Regression", best_lr, X_test_scaled),
    ("Decision Tree",       best_dt, X_test),
    ("Random Forest",       best_rf, X_test),
]:
    probs = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - All Models")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 7. MODEL SELECTION
# ============================================================
best_model_name = results.loc[results["F1 Score"].idxmax(), "Model"]
best_idx        = results["F1 Score"].idxmax()

print(f"Selected model: {best_model_name}")
print("\nFull classification report:")
print(classification_report(y_test, predictions[best_idx], target_names=["No Disease", "Disease"]))

# ============================================================
# 8. INFERENCE ON UNSEEN DATA
# ============================================================
new_patient = pd.DataFrame(
    [[21, 1, 0, 120, 200, 0, 1, 150, 0, 1.0, 2, 0, 2]],
    columns=X.columns
)

if best_model_name == "Logistic Regression":
    new_patient_input = scaler.transform(new_patient)
    best_model = best_lr
elif best_model_name == "Decision Tree":
    new_patient_input = new_patient
    best_model = best_dt
else:
    new_patient_input = new_patient
    best_model = best_rf

pred_label = best_model.predict(new_patient_input)[0]
pred_proba = best_model.predict_proba(new_patient_input)[0][1]

print(f"Prediction: {'Heart Disease Detected' if pred_label == 1 else 'No Heart Disease'}")
print(f"Confidence (probability of disease): {pred_proba:.2%}")