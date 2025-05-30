# **Employee Attrition Using Machine Learning**

A leak-proof pipeline that flags employees at risk of leaving, so HR can act early and cut turnover costs.

I refined the dataset and systematically experimented with multiple model architectures and decision thresholds, iteratively tuning hyper-parameters to identify the best-performing model for this task.

Every detail about the work is documented in the notebook.

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

## 🚀 How to Use

### 🧠 Run the Web App

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\activate   # On Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```
![Screenshot 2025-05-30 143849](https://github.com/user-attachments/assets/a63b32a7-b119-4b66-83fb-01a4047f69f1)
![Screenshot 2025-05-30 143903](https://github.com/user-attachments/assets/578236a9-ae83-4b15-b8ef-dd9f4232dd89)

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
