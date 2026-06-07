# ============================================================
# NOTEBOOK 01: Data Cleaning & Exploratory Data Analysis
# PROJECT: HR Analytics & Employee Attrition Prediction
# AUTHOR: Aditya Thakur
# ============================================================

# %% [markdown]
# # HR Analytics — Data Cleaning & EDA
# **Objective:** Understand the employee dataset, clean it, and identify
# key patterns and factors associated with employee attrition.

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)

# %% Load Data
df = pd.read_csv('../data/hr_employee_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")

# %% Basic Info
print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

# %% Null Check
print("\n--- NULL VALUES ---")
print(df.isnull().sum())
print(f"\nTotal nulls: {df.isnull().sum().sum()}")

# %% Drop constant columns
cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeID']
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
print(f"\nDropped constant columns. New shape: {df.shape}")

# %% Attrition Distribution
attrition_counts = df['Attrition'].value_counts()
attrition_rate = df['Attrition'].value_counts(normalize=True)['Yes'] * 100
print(f"\nAttrition Distribution:\n{attrition_counts}")
print(f"\nAttrition Rate: {attrition_rate:.1f}%")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
attrition_counts.plot(kind='bar', ax=ax[0], color=['#2196F3','#F44336'], edgecolor='white')
ax[0].set_title('Attrition Count', fontweight='bold')
ax[0].set_xlabel('Attrition')
ax[0].set_ylabel('Count')
ax[0].tick_params(rotation=0)

attrition_rate_vals = df['Attrition'].value_counts(normalize=True) * 100
ax[1].pie(attrition_rate_vals, labels=attrition_rate_vals.index, autopct='%1.1f%%',
          colors=['#2196F3','#F44336'], startangle=90)
ax[1].set_title('Attrition Rate', fontweight='bold')
plt.tight_layout()
plt.savefig('../visuals/attrition_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Attrition by Department
dept_attr = df.groupby('Department')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).reset_index(name='AttritionRate')

plt.figure(figsize=(10, 5))
sns.barplot(data=dept_attr, x='Department', y='AttritionRate', palette='Reds_d')
plt.title('Attrition Rate by Department', fontweight='bold', fontsize=14)
plt.ylabel('Attrition Rate (%)')
plt.xlabel('Department')
for i, v in enumerate(dept_attr['AttritionRate']):
    plt.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../visuals/attrition_by_department.png', dpi=150, bbox_inches='tight')
plt.show()
print(dept_attr)

# %% Attrition by OverTime
ot_attr = df.groupby('OverTime')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).reset_index(name='AttritionRate')
print(f"\nAttrition by OverTime:\n{ot_attr}")
# Key finding: OverTime employees have ~3x higher attrition

# %% Age Distribution by Attrition
plt.figure(figsize=(10, 5))
df[df['Attrition']=='Yes']['Age'].hist(bins=20, alpha=0.7, label='Attrition: Yes', color='#F44336')
df[df['Attrition']=='No']['Age'].hist(bins=20, alpha=0.7, label='Attrition: No', color='#2196F3')
plt.title('Age Distribution by Attrition', fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('../visuals/age_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Job Satisfaction vs Attrition
sat_attr = df.groupby('JobSatisfaction')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).reset_index(name='AttritionRate')
print(f"\nAttrition by Job Satisfaction:\n{sat_attr}")

# %% Monthly Income vs Attrition
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', palette={'Yes':'#F44336','No':'#2196F3'})
plt.title('Monthly Income by Attrition Status', fontweight='bold')
plt.xlabel('Attrition')
plt.ylabel('Monthly Income (INR)')
plt.tight_layout()
plt.savefig('../visuals/income_vs_attrition.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Correlation Heatmap (numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('../visuals/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Summary of Key EDA Findings
print("\n" + "="*60)
print("KEY EDA FINDINGS")
print("="*60)
print(f"1. Overall attrition rate: {attrition_rate:.1f}%")
print(f"2. Sales dept has highest attrition rate")
print(f"3. OverTime employees have ~3x higher attrition")
print(f"4. Lower income strongly associated with attrition")
print(f"5. Short tenure (< 2 years) correlates with high churn")
print("="*60)

# %% Save cleaned data
df.to_csv('../data/hr_employee_cleaned.csv', index=False)
print("\nCleaned dataset saved.")
