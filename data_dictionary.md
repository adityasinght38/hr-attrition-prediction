# Data Dictionary — HR Employee Dataset

## Target Variable
| Column | Values | Description |
|--------|--------|-------------|
| Attrition | Yes / No | Whether the employee left the company |

## Employee Demographics
| Column | Type | Description |
|--------|------|-------------|
| Age | Numeric | Employee age in years |
| Gender | Categorical | Male / Female |
| MaritalStatus | Categorical | Single / Married / Divorced |
| DistanceFromHome | Numeric | Distance from home to office (km) |

## Job Details
| Column | Type | Description |
|--------|------|-------------|
| Department | Categorical | Sales / R&D / Human Resources |
| JobRole | Categorical | Job title/role |
| JobLevel | Ordinal | 1 (Entry) to 5 (Executive) |
| BusinessTravel | Categorical | Non-Travel / Travel_Rarely / Travel_Frequently |
| OverTime | Binary | Yes / No |

## Compensation
| Column | Type | Description |
|--------|------|-------------|
| MonthlyIncome | Numeric | Monthly salary in INR |
| DailyRate | Numeric | Daily rate of pay |
| HourlyRate | Numeric | Hourly rate of pay |
| MonthlyRate | Numeric | Monthly rate |
| PercentSalaryHike | Numeric | % salary increase in last appraisal |
| StockOptionLevel | Ordinal | 0–4 scale |

## Satisfaction & Performance
| Column | Type | Description |
|--------|------|-------------|
| JobSatisfaction | Ordinal | 1 (Low) to 4 (Very High) |
| EnvironmentSatisfaction | Ordinal | 1 (Low) to 4 (Very High) |
| WorkLifeBalance | Ordinal | 1 (Bad) to 4 (Best) |
| JobInvolvement | Ordinal | 1 (Low) to 4 (Very High) |
| RelationshipSatisfaction | Ordinal | 1 (Low) to 4 (Very High) |
| PerformanceRating | Ordinal | 1 (Low) to 4 (Outstanding) |

## Experience & Tenure
| Column | Type | Description |
|--------|------|-------------|
| TotalWorkingYears | Numeric | Total years of work experience |
| NumCompaniesWorked | Numeric | Number of companies worked at before |
| YearsAtCompany | Numeric | Years at current company |
| YearsInCurrentRole | Numeric | Years in current role |
| YearsSinceLastPromotion | Numeric | Years since last promotion |
| YearsWithCurrManager | Numeric | Years working with current manager |
| TrainingTimesLastYear | Numeric | Number of training sessions last year |

## Education
| Column | Type | Description |
|--------|------|-------------|
| Education | Ordinal | 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor |
| EducationField | Categorical | Life Sciences / Medical / Marketing / Technical / Other |

## Notes
- Dataset: 1,470 employee records
- Attrition rate: ~23% (Yes), 77% (No) — class imbalance handled in modeling
- Constant columns dropped: EmployeeCount, StandardHours, Over18
