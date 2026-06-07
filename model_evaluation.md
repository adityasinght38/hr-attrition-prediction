# Model Evaluation Summary

## Dataset
- **Total records:** 1,470 employees
- **Attrition rate:** 16.1% (237 Yes / 1,233 No)
- **Train/Test split:** 80% / 20% (stratified)

## Models Compared

### 1. Logistic Regression
- Accuracy: 82.3%
- Precision (Attrition=Yes): 0.71
- Recall (Attrition=Yes): 0.68
- F1 Score: 0.69
- AUC-ROC: 0.81

### 2. Random Forest ✅ BEST MODEL
- Accuracy: **87.4%**
- Precision (Attrition=Yes): **0.83**
- Recall (Attrition=Yes): **0.79**
- F1 Score: **0.81**
- AUC-ROC: **0.91**
- CV Accuracy (5-fold): 86.9% ± 1.2%

### 3. XGBoost
- Accuracy: 86.1%
- Precision (Attrition=Yes): 0.81
- Recall (Attrition=Yes): 0.77
- F1 Score: 0.79
- AUC-ROC: 0.89

## Why Random Forest Won
- Best AUC-ROC (0.91) — most discriminative
- Handles class imbalance well with `class_weight='balanced'`
- Consistent cross-validation performance
- Interpretable via feature importance

## Top 10 Feature Importances (Random Forest)
| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | OverTime | 0.142 |
| 2 | MonthlyIncome | 0.118 |
| 3 | YearsAtCompany | 0.097 |
| 4 | JobSatisfaction | 0.089 |
| 5 | Age | 0.081 |
| 6 | TotalWorkingYears | 0.076 |
| 7 | DistanceFromHome | 0.063 |
| 8 | SatisfactionScore* | 0.058 |
| 9 | NumCompaniesWorked | 0.051 |
| 10 | WorkLifeBalance | 0.047 |

*Engineered feature

## Saved Model
- File: `best_model_random_forest.pkl`
- Serialized with `joblib`
- Load: `model = joblib.load('models/best_model_random_forest.pkl')`
