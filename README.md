# **Predicting Employee Attrition Using Machine Learning (Curricular Project)**

A leak-proof pipeline that flags employees at risk of leaving, so HR can act early and cut turnover costs.

---

## 📊 Dataset
| Item | Details |
|------|---------|
| **Rows** | 29 998 |
| **Numeric** | `satisfaction_level`, `last_evaluation_rating`, `projects_worked_on`, `average_monthly_hours`, `time_spend_company` |
| **Binary** | `Work_accident`, `promotion_last_5years` |
| **Categorical** | `Department`, `salary` |
| **Target** | `Attrition` (0 = Stayed, 1 = Left) |

---

🚀 Usage

1. Place dataset.csv in the project root.
2. Launch Jupyter Lab
3. Open attrition_model.ipynb and run every cell top-to-bottom.

🗂️ Project Structure

employee-attrition-ml/
├── dataset.csv
└── attrition_model.ipynb

🔑 Key Steps
1.	EDA – histograms, boxplots, count plots ⇒ class imbalance ≈ 24 % “Left”.
2.	Pre-processing
•	Stratified 70 / 20 / 10 split (train / val / test)
•	ColumnTransformer
•	StandardScaler on 5 numeric cols
•	OneHotEncoder(drop='first') on Department (9 dummies) & salary (2 dummies)
3.	Modeling & Tuning (all via Pipeline + GridSearchCV)
•	Logistic Regression (C)
•	Decision Tree (max_depth, min_leaf)
•	Random Forest (max_depth, min_leaf, n_estimators) ← best F₁ = 0.977 (val)
•	SVM-RBF (C, γ)
•	Gaussian NB (class-priors)
4.	Threshold Optimization – calibrated RF, Precision-Recall curve → best thr ≈ 0.37.
5.	Final Test Metrics (10 % hold-out)

Metric	Score
Precision (Left)	0.992
Recall (Left)	0.980
F₁ (Left)	0.986
Accuracy	0.993

📌 Findings & Recommendations
	•	Drivers: low satisfaction, 3–5 yr tenure, heavy workload, low evaluation score.
	•	Actions: targeted engagement surveys, workload caps, mid-tenure mentorship, performance coaching.

📝 License

MIT © Fady Abi Rached

