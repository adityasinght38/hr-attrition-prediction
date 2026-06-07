# ============================================================
# NOTEBOOK 02: Feature Engineering
# PROJECT: HR Analytics & Employee Attrition Prediction
# AUTHOR: Aditya Thakur
# ============================================================

# %% [markdown]
# # HR Analytics — Feature Engineering
# **Objective:** Encode categorical variables, create new meaningful features,
# and prepare the final feature matrix for model training.

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# %% Load Cleaned Data
df = pd.read_csv('../data/hr_employee_cleaned.csv')
print(f"Shape: {df.shape}")
print(f"\nCategorical columns: {df.select_dtypes(include='object').columns.tolist()}")

# %% Encode Target Variable
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
print(f"\nTarget encoded — Attrition value counts:\n{df['Attrition'].value_counts()}")

# %% Encode Binary Columns
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# %% One-Hot Encode Multi-Class Categoricals
ohe_cols = ['Department', 'EducationField', 'JobRole', 'MaritalStatus', 'BusinessTravel']
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
print(f"\nAfter one-hot encoding — Shape: {df.shape}")

# %% Feature Engineering — Create New Features
# 1. Tenure Ratio: How much of total career spent at current company
df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)

# 2. Income Per Year of Experience
df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)

# 3. Composite Satisfaction Score
df['SatisfactionScore'] = (
    df['JobSatisfaction'] +
    df['EnvironmentSatisfaction'] +
    df['WorkLifeBalance']
) / 3

# 4. Promotion Gap: Years since last promotion relative to tenure
df['PromotionGap'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)

# 5. Manager Stability: Years with current manager relative to tenure
df['ManagerStability'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)

# 6. Early Career Flag: Less than 2 years at company
df['EarlyCareer'] = (df['YearsAtCompany'] < 2).astype(int)

# 7. Low Income Flag: Below 30,000 monthly
df['LowIncome'] = (df['MonthlyIncome'] < 30000).astype(int)

print("\nNew features created:")
new_features = ['TenureRatio','IncomePerYear','SatisfactionScore',
                'PromotionGap','ManagerStability','EarlyCareer','LowIncome']
print(df[new_features].describe().round(2))

# %% Correlation of New Features with Attrition
print("\nCorrelation of engineered features with Attrition:")
print(df[new_features + ['Attrition']].corr()['Attrition'].sort_values())

# %% Visualize SatisfactionScore vs Attrition
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Attrition', y='SatisfactionScore',
            palette={1: '#F44336', 0: '#2196F3'})
plt.title('Composite Satisfaction Score vs Attrition', fontweight='bold')
plt.xlabel('Attrition (1=Yes, 0=No)')
plt.ylabel('Satisfaction Score')
plt.tight_layout()
plt.savefig('../visuals/satisfaction_vs_attrition.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Feature/Target Split
X = df.drop('Attrition', axis=1)
y = df['Attrition']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Attrition rate: {y.mean()*100:.1f}%")

# %% Scale Numeric Features
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"\nScaling applied to {len(numeric_cols)} numeric features.")

# %% Save Processed Data
X.to_csv('../data/X_features.csv', index=False)
y.to_csv('../data/y_target.csv', index=False)
X_scaled.to_csv('../data/X_scaled.csv', index=False)

print("\nFeature files saved:")
print("  → data/X_features.csv")
print("  → data/y_target.csv")
print("  → data/X_scaled.csv")
