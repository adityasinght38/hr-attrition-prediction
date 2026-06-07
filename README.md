# 🧑‍💼 HR Analytics & Employee Attrition Prediction
**Python | Pandas | Scikit-learn | Power BI | Machine Learning**

![Attrition Dashboard](reports/attrition_dashboard_preview.png)

## 🔍 Project Overview

An end-to-end HR analytics project that processes **employee workforce data**, identifies the **key drivers of attrition**, and builds **machine learning models** to predict employee turnover with **85%+ accuracy**. Accompanied by a Power BI HR dashboard for workforce planning and retention strategy.

---

## 🎯 Business Problem

High employee attrition is costly — replacing one employee can cost 50–200% of their annual salary. HR teams needed:
- Early identification of employees likely to leave
- Understanding of which factors drive attrition most
- Visual dashboards for workforce planning and retention monitoring

---

## 📁 Project Structure

```
hr-attrition-prediction/
│
├── data/
│   ├── hr_employee_data.csv            # Raw employee dataset (1,470 records)
│   └── data_dictionary.md             # Feature descriptions
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb     # Data cleaning + Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature creation and encoding
│   └── 03_model_building.ipynb       # ML models + evaluation
│
├── models/
│   ├── best_model_random_forest.pkl  # Saved best model
│   └── model_evaluation.md          # Performance metrics summary
│
├── visuals/
│   ├── attrition_by_department.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── roc_curve.png
│
└── reports/
    ├── attrition_dashboard_preview.png
    └── hr_analytics_summary.md
```

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **Pandas & NumPy** | Data cleaning, transformation, feature engineering |
| **Scikit-learn** | ML models — Logistic Regression, Random Forest, XGBoost |
| **Matplotlib & Seaborn** | EDA visualizations, feature importance plots |
| **Power BI** | HR analytics dashboard for business stakeholders |

---

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Employee age |
| Department | Categorical | HR / Sales / R&D |
| DistanceFromHome | Numeric | Commute distance (km) |
| Education | Ordinal | 1–5 scale |
| EnvironmentSatisfaction | Ordinal | 1–4 scale |
| JobInvolvement | Ordinal | 1–4 scale |
| JobLevel | Ordinal | 1–5 scale |
| JobSatisfaction | Ordinal | 1–4 scale |
| MonthlyIncome | Numeric | Monthly salary (INR) |
| NumCompaniesWorked | Numeric | Prior employers |
| OverTime | Binary | Yes / No |
| PercentSalaryHike | Numeric | Last raise % |
| TotalWorkingYears | Numeric | Total experience |
| WorkLifeBalance | Ordinal | 1–4 scale |
| YearsAtCompany | Numeric | Tenure at current company |
| **Attrition** | **Target** | **Yes / No** |

---

## 🔬 EDA Key Findings

- Employees working **OverTime** had 3× higher attrition rate (30% vs 10%)
- **Sales** department had the highest attrition (21%) vs R&D (14%)
- Employees with **low job satisfaction (score 1)** had 23% attrition rate
- **Single employees** and those with **short tenure (< 2 years)** churned most
- Attrition correlated negatively with **MonthlyIncome** and **YearsAtCompany**

---

## 🤖 Machine Learning Models

### Models Evaluated

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 82.3% | 0.71 | 0.68 | 0.69 | 0.81 |
| Random Forest | **87.4%** | **0.83** | **0.79** | **0.81** | **0.91** |
| XGBoost | 86.1% | 0.81 | 0.77 | 0.79 | 0.89 |

✅ **Best Model: Random Forest** — 87.4% accuracy, AUC-ROC 0.91

### Top 5 Features Driving Attrition (Feature Importance)
1. **OverTime** — strongest single predictor
2. **MonthlyIncome** — lower income = higher risk
3. **YearsAtCompany** — short tenure = higher risk
4. **JobSatisfaction** — low satisfaction = higher risk
5. **Age** — younger employees churn more

---

## 💻 Code Highlights

### Data Cleaning & EDA
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/hr_employee_data.csv')
print(f"Shape: {df.shape}")
print(f"Attrition rate: {df['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%")

# Check nulls
print(df.isnull().sum())

# Drop constant/irrelevant columns
df.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True)

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Attrition rate by department
dept_attrition = df.groupby('Department')['Attrition'].mean() * 100
print(dept_attrition.sort_values(ascending=False))
```

### Feature Engineering
```python
from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
cat_cols = ['Department', 'EducationField', 'Gender', 'JobRole',
            'MaritalStatus', 'OverTime', 'BusinessTravel']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Create new features
df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
df['SatisfactionScore'] = (df['JobSatisfaction'] + 
                            df['EnvironmentSatisfaction'] + 
                            df['WorkLifeBalance']) / 3
```

### Model Building & Evaluation
```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Feature/target split
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

# Fit and evaluate
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.3f}")

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind='barh', figsize=(10,6))
plt.title('Top 10 Feature Importances — Random Forest')
plt.tight_layout()
plt.savefig('visuals/feature_importance.png', dpi=150)
```

---

## 📈 Power BI Dashboard Pages

| Page | KPIs Tracked |
|------|-------------|
| **Overview** | Total employees, attrition rate, headcount trend |
| **Attrition Analysis** | By department, job role, age group, tenure |
| **Risk Scoring** | ML-predicted high-risk employees by department |
| **Retention Insights** | Satisfaction scores, salary bands, overtime impact |

---

## ⚙️ How to Run

```bash
# Clone the repo
git clone https://github.com/adityasinght38/hr-attrition-prediction.git
cd hr-attrition-prediction

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_data_cleaning_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_model_building.ipynb
```

---

## 📦 Requirements

```
pandas==2.1.0
numpy==1.25.2
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
xgboost==1.7.6
imbalanced-learn==0.11.0
joblib==1.3.2
jupyter==1.0.0
```

---

## 📌 Results & Impact

| Metric | Result |
|--------|--------|
| Dataset Size | 1,470 employee records |
| Best Model Accuracy | **87.4%** (Random Forest) |
| AUC-ROC Score | **0.91** |
| Top Attrition Driver | OverTime (3× higher risk) |
| High-Risk Employees Identified | 18% of workforce |
| Dashboard Pages | 4 pages, 15+ KPIs |

---

## 👤 Author

**Aditya Thakur**  
[LinkedIn](https://www.linkedin.com/in/aditya-singh-thakur-010913298/) • [GitHub](https://github.com/adityasinght38)  
B.Tech Computer Science Engineering | VIT Vellore
