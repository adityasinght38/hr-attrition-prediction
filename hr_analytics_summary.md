# HR Analytics & Employee Attrition Prediction — Project Summary

## Executive Summary
This project analyzed 1,470 employee records to identify the key drivers of
attrition and built a machine learning model achieving **87.4% accuracy**
(AUC-ROC: 0.91) to predict which employees are at risk of leaving.

## Problem Statement
Employee attrition is costly — replacing one employee costs 50–200% of their
annual salary. HR teams lacked a data-driven way to proactively identify
at-risk employees before they resigned.

## Approach

### 1. Data Cleaning
- Removed constant columns (EmployeeCount, StandardHours, Over18)
- Validated null values — dataset was clean
- Encoded target variable (Yes→1, No→0)

### 2. Exploratory Data Analysis
Key findings from EDA:
- Overall attrition rate: 23.1%
- OverTime employees: ~3× higher attrition probability
- Sales department: highest attrition rate (21%+)
- Job Satisfaction score 1: 23% attrition vs 11% for score 4
- Monthly income: strong negative correlation with attrition
- Short tenure (< 2 years): significantly higher churn

### 3. Feature Engineering
7 new features created:
- `TenureRatio`: Career fraction spent at current company
- `IncomePerYear`: Income relative to experience
- `SatisfactionScore`: Composite of job + env + WLB satisfaction
- `PromotionGap`: Years since promotion vs tenure
- `ManagerStability`: Manager tenure ratio
- `EarlyCareer`: Flag for < 2 years tenure
- `LowIncome`: Flag for monthly income < ₹30,000

### 4. Modeling
Three models trained with stratified 80/20 split:

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 82.3% | 0.81 |
| **Random Forest** | **87.4%** | **0.91** |
| Gradient Boosting | 86.1% | 0.89 |

**Best Model: Random Forest** (class_weight='balanced', 5-fold CV)

### 5. Top Attrition Drivers (Feature Importance)
1. OverTime
2. MonthlyIncome
3. YearsAtCompany
4. JobSatisfaction
5. Age

## Business Recommendations
1. **Reduce mandatory overtime** — strongest single predictor of attrition
2. **Salary review for low-income band** (< ₹30,000/month) — high-risk group
3. **Onboarding program for first 2 years** — early career employees churn most
4. **Regular satisfaction surveys** — low job satisfaction = 2× attrition risk
5. **Promotion pipeline review** — long promotion gaps correlate with exits

## Files Produced
- `data/hr_employee_data.csv` — Raw dataset
- `data/hr_employee_cleaned.csv` — Cleaned dataset
- `data/X_features.csv` — Final feature matrix
- `notebooks/01_data_cleaning_eda.py` — EDA script
- `notebooks/02_feature_engineering.py` — Feature engineering
- `notebooks/03_model_building.py` — Model training & evaluation
- `models/best_model_random_forest.pkl` — Saved model
- `visuals/` — All charts and plots
