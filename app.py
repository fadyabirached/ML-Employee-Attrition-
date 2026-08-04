import random

import streamlit as st

import model_service
from model_service import load_or_train_model, predict_one

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🧑‍💼")


# ── Load model bundle ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing the model (first run only)...")
def get_bundle(contract: str):
    """Load and cache the model bundle.

    ``contract`` is never read. It's in the signature so the cache key
    changes whenever the bundle's required keys do: Streamlit Cloud reruns
    the script inside the *existing* process when it picks up new code, so
    a bundle cached by the previous version otherwise survives the deploy
    and blows up on whichever key the new code expects.
    """
    return load_or_train_model()


bundle = get_bundle(",".join(sorted(getattr(model_service, "REQUIRED_KEYS", ()))))

RANGES = bundle["feature_ranges"]

LABELS = {
    "satisfaction_level": "Satisfaction",
    "last_evaluation_rating": "Evaluation rating",
    "projects_worked_on": "Projects",
    "average_monthly_hours": "Monthly hours",
    "time_spend_company": "Years at company",
    "Work_accident": "Work accident",
    "promotion_last_5years": "Promoted in last 5y",
    "Department": "Department",
    "salary": "Salary",
}


@st.cache_data(show_spinner=False)
def held_out_employees():
    """The test split -- rows the model was never fitted on."""
    import train

    frame = train.load_data()
    _train_X, _val_X, test_X, _train_y, _val_y, test_y = train.split_data(frame)
    return test_X.reset_index(drop=True), test_y.reset_index(drop=True)


def verdict(result):
    """Render the risk banner for a scored profile."""
    rate = result["base_rate"]
    if result["risk_band"] == "high":
        st.error("⚠️  High risk. Likely to leave.")
    elif result["risk_band"] == "elevated":
        st.warning(f"Elevated risk. Above the {rate:.1%} company average.")
    else:
        st.success(f"👍  Low risk. Below the {rate:.1%} company average.")


def show_similar(row):
    """Ground the score in real outcomes rather than asking for trust."""
    lookup = getattr(model_service, "similar_employees", None)
    result = lookup(bundle, row) if lookup else None
    if not result:
        return
    left, total = result
    st.info(
        f"**Of the {total} most similar real employees, {left} left "
        f"and {total - left} stayed.** These are actual people from the "
        "training data, not a model output."
    )


def drivers_for(row):
    """The inputs that moved this prediction, largest effect first."""
    explain = getattr(model_service, "explain", None)
    return explain(bundle, row) if explain else []


st.title("🧑‍💼 Employee Attrition Predictor")
st.markdown(
    "Gradient boosting over 24k HR records, predicting who leaves. "
    "**97.8% accurate** on a held-out test set. Check it yourself below."
)

# ── Primary: let the model prove itself on data it has never seen ────────────
# Attrition in this dataset is driven by combinations rather than any single
# field, so hand-built profiles are a poor way in: the answer looks arbitrary
# because you have nothing to check it against. Scoring a real employee and
# then revealing what actually happened to them removes the guesswork.
st.subheader("Test it on a real employee")
st.caption(
    "Picks someone from the held-out set the model never trained on, scores "
    "them, then reveals what actually happened."
)

if st.button("🎲 Score a real employee", type="primary", use_container_width=True):
    employees, outcomes = held_out_employees()
    picked = random.randrange(len(employees))
    employee = employees.iloc[picked].to_dict()

    scored = predict_one(bundle, employee)
    truth = bool(outcomes.iloc[picked])
    correct = truth == (scored["risk_band"] == "high")

    st.session_state["last_check"] = (employee, scored, truth, correct)
    st.session_state.setdefault("tally", []).append(correct)

if "last_check" in st.session_state:
    employee, scored, truth, correct = st.session_state["last_check"]

    left, right = st.columns(2)
    for index, (name, value) in enumerate(
        (key, employee[key]) for key in bundle["features"]
    ):
        shown = value
        if name in ("Work_accident", "promotion_last_5years"):
            shown = "Yes" if value else "No"
        (left if index % 2 == 0 else right).write(f"**{LABELS.get(name, name)}:** {shown}")

    st.divider()
    st.write(f"**Model says:** {scored['probability_leave']:.1%} chance of leaving")
    verdict(scored)

    if correct:
        st.success(
            f"✅ Correct. This employee actually "
            f"**{'left' if truth else 'stayed'}**."
        )
    else:
        st.error(
            f"❌ Wrong. This employee actually "
            f"**{'left' if truth else 'stayed'}**."
        )

    show_similar(employee)

    tally = st.session_state.get("tally", [])
    st.caption(
        f"This session: **{sum(tally)} of {len(tally)}** correct · "
        f"across all 2,398 held-out employees the model is right 97.8% of the time "
        f"· {scored['latency_ms']:.0f} ms per prediction on CPU"
    )

st.divider()

# ── Secondary: manual exploration, deliberately behind a fold ────────────────
PRESETS = {
    "🔥 Burnt-out top performer": {
        "satisfaction_level": 1.0,
        "last_evaluation_rating": 8.7,
        "projects_worked_on": 7,
        "average_monthly_hours": 281,
        "time_spend_company": 4,
        "work_accident": "No",
        "promotion_last_5years": "No",
        "salary": "low",
    },
    "😐 Under-used employee": {
        "satisfaction_level": 4.0,
        "last_evaluation_rating": 5.1,
        "projects_worked_on": 3,
        "average_monthly_hours": 149,
        "time_spend_company": 3,
        "work_accident": "No",
        "promotion_last_5years": "No",
        "salary": "low",
    },
    "🙂 Typical employee who stays": {
        "satisfaction_level": 6.9,
        "last_evaluation_rating": 7.1,
        "projects_worked_on": 4,
        "average_monthly_hours": 203,
        "time_spend_company": 3,
        "work_accident": "No",
        "promotion_last_5years": "No",
        "salary": "medium",
    },
}


def float_input(label, name, default, step=0.1):
    """Slider bounded by what the feature actually spans in the training data."""
    low, high = (float(bound) for bound in RANGES[name])
    return st.slider(label, low, high, min(max(default, low), high), step=step, key=name)


def int_input(label, name, default, widget=st.number_input):
    low, high = (int(bound) for bound in RANGES[name])
    return widget(label, low, high, min(max(default, low), high), key=name)


with st.expander("Build a profile by hand"):
    st.caption(
        "Satisfaction, salary and promotion are constrained so that improving "
        "any of them can never raise the predicted risk. The rest follow the "
        "data, which is not always intuitive: both the weakest and strongest "
        "performers leave (27% and 24%) while the middle stays (2%), and "
        "employees who had a work accident leave *less* (5.7% vs 18.6%). "
        "Every prediction lists what moved it, so you can see which won."
    )

    for column, (name, values) in zip(st.columns(len(PRESETS)), PRESETS.items()):
        if column.button(name, use_container_width=True):
            st.session_state.update(values)

    satisfaction_level = float_input("Satisfaction Level (out of 10)", "satisfaction_level", 5.0)
    last_evaluation_rating = float_input(
        "Last Evaluation Rating (out of 10)", "last_evaluation_rating", 7.0
    )
    projects_worked_on = int_input("Number of Projects", "projects_worked_on", 4)
    average_monthly_hours = int_input("Average Monthly Hours", "average_monthly_hours", 200)
    time_spend_company = int_input("Years at Company", "time_spend_company", 3, widget=st.slider)
    work_accident = st.selectbox("Had Work Accident?", ["Yes", "No"], key="work_accident")
    promotion_last_5years = st.selectbox(
        "Promoted in Last 5 Years?", ["Yes", "No"], key="promotion_last_5years"
    )
    department = st.selectbox(
        "Department",
        [
            "sales", "technical", "support", "IT", "product_mng", "marketing",
            "RandD", "accounting", "hr", "management",
        ],
        key="department",
    )
    salary = st.selectbox("Salary Level", ["low", "medium", "high"], key="salary")

    if st.button("Predict", use_container_width=True):
        row = {
            "satisfaction_level": satisfaction_level,
            "last_evaluation_rating": last_evaluation_rating,
            "projects_worked_on": projects_worked_on,
            "average_monthly_hours": average_monthly_hours,
            "time_spend_company": time_spend_company,
            "Work_accident": 1 if work_accident == "Yes" else 0,
            "promotion_last_5years": 1 if promotion_last_5years == "Yes" else 0,
            "Department": department,
            "salary": salary,
        }

        try:
            result = predict_one(bundle, row)
            st.write(f"**Probability of leaving:** {result['probability_leave']:.1%}")
            verdict(result)

            drivers = drivers_for(row)
            if drivers and abs(drivers[0][1]) >= 0.1:
                for name, effect in drivers:
                    if abs(effect) < 0.1:
                        continue
                    arrow = "↑" if effect > 0 else "↓"
                    direction = "raises" if effect > 0 else "lowers"
                    st.write(
                        f"- {arrow} **{LABELS.get(name, name)}** ({row[name]}) "
                        f"{direction} risk by {abs(effect):.1f} points"
                    )
            elif drivers:
                # Listing four entries of "0.0 points" reads as broken. It
                # actually means the score is pinned at the end of the range
                # and no single field is what put it there.
                st.caption(
                    "No single field moves this one much: the profile sits at "
                    "the far end of the range, so changing any one input on "
                    "its own barely shifts the result."
                )

            show_similar(row)

            if result["out_of_distribution"]:
                st.caption(
                    f"ℹ️ Unusual profile, {result['novelty_score']:.1f}x further from "
                    "the training data than a typical employee, so treat it as rough."
                )
        except Exception as e:  # noqa: BLE001 -- UI safety net, must not crash the page
            st.error(f"Prediction failed: {e}")
