# ============================================================
# NOTEBOOK 03: Model Building & Evaluation
# PROJECT: HR Analytics & Employee Attrition Prediction
# AUTHOR: Aditya Thakur
# ============================================================

# %% [markdown]
# # HR Analytics — Model Building & Evaluation
# **Objective:** Train multiple classification models to predict employee attrition,
# evaluate performance, and select the best model for deployment.

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score,
                              precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler

sns.set_theme(style='whitegrid', palette='muted')

# %% Load Data
X = pd.read_csv('../data/X_features.csv')
y = pd.read_csv('../data/y_target.csv').squeeze()

print(f"Features: {X.shape[1]} | Records: {X.shape[0]}")
print(f"Attrition rate: {y.mean()*100:.1f}%")

# %% Train/Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train attrition rate: {y_train.mean()*100:.1f}%")
print(f"Test  attrition rate: {y_test.mean()*100:.1f}%")

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# %% ─── MODEL 1: LOGISTIC REGRESSION ─────────────────────────
print("\n" + "="*50)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*50)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
y_prob_lr = lr.predict_proba(X_test_s)[:, 1]

print(classification_report(y_test, y_pred_lr, target_names=['No Attrition','Attrition']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_lr):.3f}")

# %% ─── MODEL 2: RANDOM FOREST ────────────────────────────────
print("\n" + "="*50)
print("MODEL 2: RANDOM FOREST")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_rf, target_names=['No Attrition','Attrition']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.3f}")

# %% ─── MODEL 3: GRADIENT BOOSTING ───────────────────────────
print("\n" + "="*50)
print("MODEL 3: GRADIENT BOOSTING (XGBoost-style)")
print("="*50)

gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
y_prob_gb = gb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_gb, target_names=['No Attrition','Attrition']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_gb):.3f}")

# %% Model Comparison Table
results = {
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_gb)
    ],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_gb)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_gb)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_gb)
    ],
    'AUC-ROC': [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_gb)
    ]
}
results_df = pd.DataFrame(results)
results_df[['Accuracy','Precision','Recall','F1 Score','AUC-ROC']] = \
    results_df[['Accuracy','Precision','Recall','F1 Score','AUC-ROC']].round(3)

print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))

# %% ROC Curve — All Models
plt.figure(figsize=(10, 7))
for name, y_prob in [('Logistic Regression', y_prob_lr),
                      ('Random Forest', y_prob_rf),
                      ('Gradient Boosting', y_prob_gb)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve — Model Comparison', fontweight='bold', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../visuals/roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Confusion Matrix — Best Model (Random Forest)
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Attrition','Attrition'],
            yticklabels=['No Attrition','Attrition'])
plt.title('Confusion Matrix — Random Forest', fontweight='bold', fontsize=13)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('../visuals/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Feature Importance — Random Forest
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
top_features = feat_imp.nlargest(15)

plt.figure(figsize=(10, 7))
top_features.sort_values().plot(kind='barh', color='#1B3A6B', edgecolor='white')
plt.title('Top 15 Feature Importances — Random Forest', fontweight='bold', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('../visuals/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 10 Features:")
print(feat_imp.nlargest(10))

# %% Save Best Model
joblib.dump(rf, '../models/best_model_random_forest.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
print("\n✅ Best model saved: models/best_model_random_forest.pkl")
print("✅ Scaler saved: models/scaler.pkl")

# %% Predict on New Employee (Example Inference)
print("\n--- EXAMPLE: PREDICT ATTRITION RISK FOR NEW EMPLOYEE ---")
# Load model
model = joblib.load('../models/best_model_random_forest.pkl')

# Sample employee (same feature order as training)
sample = X_test.iloc[[0]]
prob = model.predict_proba(sample)[0][1]
pred = model.predict(sample)[0]
print(f"Predicted Attrition: {'YES ⚠️' if pred == 1 else 'NO ✅'}")
print(f"Attrition Probability: {prob*100:.1f}%")

print("\n" + "="*50)
print("✅ BEST MODEL: Random Forest")
print(f"   Accuracy  : {accuracy_score(y_test, y_pred_rf)*100:.1f}%")
print(f"   AUC-ROC   : {roc_auc_score(y_test, y_prob_rf):.3f}")
print(f"   F1 Score  : {f1_score(y_test, y_pred_rf):.3f}")
print("="*50)
