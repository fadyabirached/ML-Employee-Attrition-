import joblib
import pandas as pd
import streamlit as st
import time

# ── Load model bundle ────────────────────────────────────────────────────────
bundle      = joblib.load("employee_attrition_rf.pkl")
model       = bundle["model"]
best_thr    = bundle["threshold"]
feat_cols   = bundle["features"]

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🧑‍💼 Employee Attrition Predictor")

st.markdown(
    "Fill in the employee's details and click **Predict**. "
    "The model returns the probability they will leave the company."
)

# ── Input Fields (Match training features) ───────────────────────────────────
satisfaction_level      = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, step=0.01)
last_evaluation_rating  = st.slider("Last Evaluation Rating", 0.0, 1.0, 0.5, step=0.01)
projects_worked_on      = st.number_input("Number of Projects", 1, 10, 3)
average_monthly_hours   = st.number_input("Average Monthly Hours", 50, 350, 160)
time_spend_company      = st.slider("Years at Company", 1, 10, 3)
work_accident           = st.selectbox("Had Work Accident?", ["Yes", "No"])
promotion_last_5years   = st.selectbox("Promoted in Last 5 Years?", ["Yes", "No"])
department              = st.selectbox("Department", [
    "sales", "technical", "support", "IT", "product_mng", "marketing",
    "RandD", "accounting", "hr", "management"
])
salary                  = st.selectbox("Salary Level", ["low", "medium", "high"])

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("Predict"):
    row_dict = {
        "satisfaction_level": satisfaction_level,
        "last_evaluation_rating": last_evaluation_rating,
        "projects_worked_on": projects_worked_on,
        "average_monthly_hours": average_monthly_hours,
        "time_spend_company": time_spend_company,
        "Work_accident": 1 if work_accident == "Yes" else 0,
        "promotion_last_5years": 1 if promotion_last_5years == "Yes" else 0,
        "Department": department,
        "salary": salary
    }

    row = pd.DataFrame([row_dict])

    # Ensure correct dtypes
    row["Department"] = row["Department"].astype(str)
    row["salary"] = row["salary"].astype(str)

    # Match training feature order
    row = row.reindex(columns=feat_cols)

    try:
        # Optional warm-up (avoids first-call overhead in timing)
        _ = model.predict_proba(row)

        # Measure inference latency
        start = time.perf_counter()
        prob_leave = float(model.predict_proba(row)[:, 1][0])
        lat_ms = (time.perf_counter() - start) * 1000

        will_leave = int(prob_leave >= best_thr)

        st.write(f"**Probability of leaving:** {prob_leave:.1%}")
        st.caption(f"Inference latency (CPU, batch=1): {lat_ms:.2f} ms")

        if will_leave:
            st.error("⚠️  High risk - likely to leave.")
        else:
            st.success("👍  Low risk - likely to stay.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Debug info:")
        st.dataframe(row)
        st.write(row.dtypes)
