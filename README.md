# 🧑‍💼 Employee Attrition Predictor

An end-to-end ML pipeline that flags employees at risk of leaving, so HR can act early and cut turnover costs. Covers EDA, a leak-free preprocessing/training pipeline, model comparison, threshold tuning, and two ways to serve the result: a Streamlit demo and a JSON API.

**[Try the live demo →](https://employee-hr-attrition.streamlit.app/)**

---

## 📊 Dataset

| Item | Details |
|------|---------|
| **Rows (raw / deduplicated)** | 29 998 / 23 980 |
| **Numeric** | `satisfaction_level`, `last_evaluation_rating`, `projects_worked_on`, `average_monthly_hours`, `time_spend_company` |
| **Binary** | `Work_accident`, `promotion_last_5years` |
| **Categorical** | `Department` (one-hot), `salary` (ordinal: low < medium < high) |
| **Target** | `Attrition` (0 = Stayed, 1 = Left) |

### Data quality: duplicate rows

~20% of the raw rows (6 018 of 29 998) are **exact duplicates**. Left in place, a random train/val/test split lets identical rows land in both the training set and the test set. The model can "recognize" a test row it memorized during training instead of generalizing to it. This is a distinct leak from (and in addition to) fitting the `ColumnTransformer` only on the training fold.

The pipeline now drops exact duplicates *before* splitting (`train.load_data()`, and the equivalent cell at the top of the notebook). This measurably changes the reported numbers below versus a naive fit-on-everything approach. The metrics here are post-fix and should be the ones you trust.

---

## 🔑 Pipeline

1. **EDA**: histograms, boxplots, count plots ⇒ class imbalance ≈ 24% "Left" in the raw rows, 16.6% once the duplicates above are dropped.
2. **Preprocessing**
   - Deduplicate, then stratified 70 / 20 / 10 split (train / val / test)
   - `ColumnTransformer`: `StandardScaler` on 5 numeric columns, `OrdinalEncoder` on `salary` (low < medium < high), `OneHotEncoder(drop='first')` on `Department` (9 dummies), and the two 0/1 flags passed through, fit on the training fold only
   - The flags need naming explicitly: `ColumnTransformer` defaults to `remainder='drop'`, so `Work_accident` and `promotion_last_5years` were being discarded despite a 3-4x difference in attrition rate across each
3. **Modeling & tuning** (`Pipeline` + `GridSearchCV`, 5-fold stratified CV, scored on F1 for the "Left" class)
   - The notebook compares five classical models: Logistic Regression, Decision Tree, Random Forest, SVM-RBF and Gaussian NB. Random Forest scored best.
   - `train.py` ships **`HistGradientBoostingClassifier`** instead (CV F1 = 0.926, `learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=20`). It was not part of that comparison; it replaced the Random Forest later because it supports monotonic constraints, and none of the five did what was needed on that front:

   | Model | Test F1 | Improving an input can never raise risk |
   |---|---|---|
   | Random Forest (notebook winner) | 0.954 | no |
   | **Gradient boosting + constraints (shipped)** | **0.930** | **yes** |
   | Logistic Regression | 0.528 | yes |

   See [Constraints](#-constraining-what-the-model-is-allowed-to-learn) for why that trade is worth two points of F1.
4. **Threshold optimization**: precision/recall curve on the validation set → best threshold ≈ 0.829
5. **Final test-set metrics** (10% hold-out, corrected for the duplicate-row issue above)

| Metric | Score |
|---|---|
| Precision (Left) | 0.957 |
| Recall (Left) | 0.905 |
| F1 (Left) | 0.930 |
| Accuracy | 0.978 |

The EDA, the five-model comparison and the plots live in `ml-employee-attrition.ipynb`, which is the original exploration. `train.py` is the current pipeline and has since moved past it (constrained gradient boosting rather than the notebook's Random Forest); it is a plain script with no notebook required, and it is what the test suite, CI and both serving interfaces use.

📌 **Findings**
- The two groups that leave are a burnt-out top performer (satisfaction 1.0, evaluation 8.7, 7 projects, 281 h/month, 4 years) and an under-used one (4.0, 5.1, 3 projects, 149 h/month, 3 years). Neither is reachable by moving a single factor.
- Evaluation rating is U-shaped: the weakest and the strongest performers leave (27% and 24%) while the middle stays (2%). Low salary roughly quadruples the rate against high (20.4% vs 4.8%), and a promotion in the last five years cuts it from 16.8% to 3.9%.
- Actions those point to: workload caps for high performers, real projects for the under-used, and pay and promotion review at the low end.

---

## 🛠️ Tech Stack

- **Modeling:** scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`, `HistGradientBoostingClassifier` with monotonic constraints)
- **Data:** pandas, numpy
- **Serving:** Streamlit (`app.py`) and FastAPI (`api.py`), two interfaces over one shared prediction path
- **Validation:** Pydantic (request schema + range validation on the API)
- **Testing / CI:** pytest, ruff, GitHub Actions

---

## 🧠 Two ways to serve the model

- **`app.py`**: Streamlit form for interactive use.
- **`api.py`**: FastAPI `/predict` endpoint (JSON in, probability out), with request validation and a `/health` check.

Both share one implementation in `model_service.py`. The trained model isn't committed to git; each interface loads it from disk if present, or auto-trains a fresh one (~8s) if not. Run `python train.py` to persist it locally.

---

## 🔒 Constraining what the model is allowed to learn

Attrition in this dataset is genuinely non-monotonic. Both the weakest and the strongest performers leave (27% and 24%) while the middle stays (2%), and employees with a recorded work accident leave *less* (5.7% vs 18.6%). An unconstrained model fits that faithfully and scores well, but it will happily report a **more** satisfied, better paid employee as **more** likely to leave, which is impossible to defend to a user no matter what the accuracy is.

So three relationships are enforced structurally, via `monotonic_cst` on the gradient booster:

| Constraint | Why |
|---|---|
| higher satisfaction never raises risk | unambiguous, and matches the data overall |
| higher salary never raises risk | 20.4% low, 14.6% medium, 4.8% high |
| a recent promotion never raises risk | 16.8% without, 3.9% with |

`salary` is ordinal-encoded rather than one-hot for this: as separate dummies nothing forces low ≥ medium, and that ordering was violated on 24 of 200 random profiles before the change.

Evaluation rating and work accident are deliberately left free. Their real relationships contradict intuition, and constraining them would mean forcing the model to disagree with the data rather than encoding domain knowledge.

**What it costs:** F1 0.930 against 0.954 unconstrained. For comparison, Logistic Regression is monotonic by construction but scores 0.528. Verified over 150 random profiles through the serving path: zero violations on all three constraints.

---

## 🛡️ Not answering questions it can't answer

A tree model returns a probability for *any* input, so three things keep the demo from dressing up a guess as a result:

- **Inputs are bounded by the training data's real ranges** (satisfaction 0.9 to 10, evaluation 3.6 to 10, tenure 2 to 10 years), taken from the data at train time rather than hardcoded, so you cannot ask about an employee who never existed.
- **Profiles unlike anything in the data get the estimate plus a caveat.** Per-feature bounds can't catch this: every value can be in range while the *combination* never occurs, since the low-satisfaction employees here are all long-tenured high performers. A nearest-neighbour check against the training set (`train.build_ood_detector`) scores how far out a request sits. It annotates rather than refuses, and the bar is 2x the in-data limit, not 1x, because the data occupies a thin manifold inside its own bounding box: 60% of randomly sampled in-range profiles exceed 1x, against 2% at 2x.
- **Every prediction is shown next to real outcomes.** The nearest-neighbour index doubles as a lookup for what actually happened to the 15 most similar employees in the training data. A score that contradicts intuition is far easier to judge next to "1 of 15 comparable people left", and it is the one claim on the page that does not require trusting the model at all.
- **Results are banded against the 16.6% company-wide rate**, so a probability that merely matches the base rate reads as "no clear signal" rather than a reassuring "likely to stay".

---

## 🚀 How to use

### Retrain the model

```bash
pip install -r requirements.txt
python train.py
```

Loads `dataset.csv`, deduplicates, grid-searches the model, tunes the threshold on validation, evaluates on the held-out test set, and writes `employee_attrition_model.pkl`.

### Run the web app

```bash
pip install -r requirements.txt
streamlit run app.py
```

![Screenshot 2025-05-30 143849](https://github.com/user-attachments/assets/a63b32a7-b119-4b66-83fb-01a4047f69f1)
![Screenshot 2025-05-30 143903](https://github.com/user-attachments/assets/578236a9-ae83-4b15-b8ef-dd9f4232dd89)

### Run the API

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`, or:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "satisfaction_level": 3.8, "last_evaluation_rating": 5.3,
  "projects_worked_on": 2, "average_monthly_hours": 157,
  "time_spend_company": 3, "Work_accident": 0,
  "promotion_last_5years": 0, "Department": "sales", "salary": "low"
}'
```

### Run the tests

```bash
pip install -r requirements.txt pytest httpx
pytest tests -v
```

CI (`.github/workflows/ci.yml`) runs lint + the full test suite (training pipeline, `model_service`, and the API) on every push/PR to `main`.

---

## 📁 Project Structure

```
employee-attrition/
├── ml-employee-attrition.ipynb   # Full EDA + model comparison + plots
├── train.py                      # Productionized winning pipeline (used by tests/CI)
├── model_service.py              # Shared load-or-train + predict logic
├── app.py                        # Streamlit UI, built on model_service
├── api.py                        # FastAPI service, built on model_service
├── dataset.csv                   # Source data
├── tests/
│   ├── test_train.py             # Leakage regression, stratification, monotonic constraints
│   ├── test_model_service.py     # predict_one, explain, risk bands, novelty guard
│   └── test_api.py               # API request validation + response shape
├── requirements.txt
├── pyproject.toml                # pytest config (puts the repo root on sys.path)
└── .github/workflows/ci.yml      # Lint + test suite on push/PR
```

---

## 📝 License

MIT © Fady Abi Rached
