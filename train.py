"""Reproducible training script for the employee attrition model.

This is the productionized version of the pipeline explored in
ml-employee-attrition.ipynb: gradient boosting with monotonic
constraints, plus threshold tuning.

Usage:
    python train.py
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC_FEATURES = [
    "satisfaction_level",
    "last_evaluation_rating",
    "projects_worked_on",
    "average_monthly_hours",
    "time_spend_company",
]
CATEGORICAL_FEATURES = ["Department"]
# Ordinal rather than one-hot so a single monotonic constraint can enforce
# low >= medium >= high. As separate dummies the ordering between low and
# medium is unconstrained, and it does get violated in practice.
ORDINAL_FEATURES = ["salary"]
SALARY_ORDER = ["low", "medium", "high"]
# Already 0/1, so they need no encoding -- but they still have to be listed.
# ColumnTransformer defaults to remainder="drop", so any column not named in
# a transformer is silently discarded: these two were reaching the fit and
# being thrown away, despite a 3-4x difference in attrition rate across each.
BINARY_FEATURES = ["Work_accident", "promotion_last_5years"]
TARGET = "Attrition"

PARAM_GRID = {
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_leaf_nodes": [31, 63],
    "clf__min_samples_leaf": [20, 50],
}

# Relationships where the business meaning is unambiguous, expressed as
# sklearn monotonic constraints (-1 decreasing, +1 increasing) on the
# transformed feature names.
#
# Deliberately partial. Evaluation rating is genuinely U-shaped here (the
# weakest and the strongest performers both leave, the middle stays) and
# employees with a recorded work accident leave less, not more, so
# constraining either would be forcing the model to contradict the data
# rather than encoding domain knowledge. These three are the ones where
# intuition and the data agree, and without them the model is free to
# produce a higher risk for a more satisfied, better paid, just-promoted
# employee, which is indefensible to a user however good the accuracy is.
MONOTONIC_CONSTRAINTS = {
    "num__satisfaction_level": -1,
    "ord__salary": -1,
    "bin__promotion_last_5years": -1,
}


def load_data(path: str = "dataset.csv") -> pd.DataFrame:
    """Load the dataset and drop exact duplicate rows.

    ~20% of the raw rows are exact duplicates. Left in place, a random
    split lets identical rows land in both train and test, which lets the
    model "recognize" a test row it memorized during training rather than
    generalize to it. Deduplicating before splitting closes that leak.
    """
    df = pd.read_csv(path)
    return df.drop_duplicates().reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("ord", OrdinalEncoder(categories=[SALARY_ORDER]), ORDINAL_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def monotonic_constraint_vector(preprocessor) -> list:
    """Line MONOTONIC_CONSTRAINTS up with the transformed column order.

    The constraint list sklearn wants is positional, so it has to be built
    from the fitted preprocessor's output names rather than written out by
    hand: adding a department or reordering a transformer would otherwise
    silently apply a constraint to the wrong column.
    """
    return [
        MONOTONIC_CONSTRAINTS.get(name, 0)
        for name in preprocessor.get_feature_names_out()
    ]


def split_data(df: pd.DataFrame, random_state: int = 42):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, param_grid=None, cv_splits: int = 5, random_state: int = 42):
    """Grid-search the gradient boosting pipeline and return the best estimator."""
    # Fitting the preprocessor up front is what makes the constraints
    # positionally correct; it is cheap next to the search itself.
    constraints = monotonic_constraint_vector(build_preprocessor().fit(X_train))
    pipe = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    monotonic_cst=constraints,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        pipe,
        param_grid or PARAM_GRID,
        scoring=make_scorer(f1_score, pos_label=1),
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_threshold(model, X_val, y_val) -> float:
    """Pick the decision threshold on the validation set that maximizes F1."""
    probs_val = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_idx = np.nanargmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def evaluate(model, threshold: float, X_test, y_test) -> str:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return classification_report(y_test, preds, target_names=["Stayed", "Left"], digits=4)


def feature_ranges(X_train) -> dict:
    """Observed min/max of each numeric feature.

    Serving interfaces use these to bound their inputs, so a user can't
    ask about an employee who works 40 hours a month or has been here
    less than the shortest tenure in the data.
    """
    return {
        column: (float(X_train[column].min()), float(X_train[column].max()))
        for column in NUMERIC_FEATURES
    }


def baseline_profile(X_train) -> dict:
    """The ordinary employee every prediction gets explained against.

    Median for the numerics, most common value for everything else. Swap
    one field of a real request for its baseline value and the change in
    probability is that field's contribution.
    """
    profile = {column: float(X_train[column].median()) for column in NUMERIC_FEATURES}
    for column in ORDINAL_FEATURES + CATEGORICAL_FEATURES:
        profile[column] = X_train[column].mode().iat[0]
    for column in BINARY_FEATURES:
        profile[column] = int(X_train[column].median())
    return profile


def build_ood_detector(model, X_train, y_train=None) -> dict:
    """Index the training rows so serving can spot inputs unlike any of them.

    A Random Forest returns a probability for any input, including feature
    combinations that never occur in the data -- rock-bottom satisfaction
    paired with a low evaluation and minimal tenure, say, when every real
    low-satisfaction employee is a long-tenured high performer. The forest
    routes such a row down whichever branches happen to fit and reports a
    number that looks like a considered answer but isn't.

    Bounding each feature separately can't catch that: every value is in
    range, only the combination is impossible. So embed the training rows
    through the fitted preprocessor and measure how far each sits from its
    nearest neighbour.

    The threshold is the 99th percentile of those distances, not the
    maximum -- the maximum is set by whichever single row happens to be
    most isolated (here 2.24 against a median of 0.44), which makes the
    guard far too permissive. p99 means "further out than all but the 1%
    most isolated training rows".
    """
    from sklearn.neighbors import NearestNeighbors

    embedded = model.named_steps["prep"].transform(X_train)
    index = NearestNeighbors(n_neighbors=2).fit(embedded)
    neighbour_distances, _ = index.kneighbors(embedded)
    # Column 0 is the row itself at distance 0; column 1 is its true neighbour.
    threshold = float(np.percentile(neighbour_distances[:, 1], 99))
    # Keeping the outcomes lets serving answer the question a probability
    # cannot: what actually happened to the people this request resembles.
    # An unintuitive score is a great deal easier to accept next to fifteen
    # real employees who did the same thing.
    outcomes = None if y_train is None else np.asarray(y_train)
    return {"index": index, "threshold": threshold, "outcomes": outcomes}


def main():
    import joblib

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    model, best_params, cv_f1 = train_model(X_train, y_train)
    print(f"Best params: {best_params}  |  CV F1: {cv_f1:.4f}")

    threshold = tune_threshold(model, X_val, y_val)
    print(f"Tuned decision threshold: {threshold:.3f}")

    print("\nTest-set performance:")
    print(evaluate(model, threshold, X_test, y_test))

    artifact = {
        "model": model,
        "threshold": threshold,
        "features": list(X_train.columns),
        "base_rate": float(y_train.mean()),
        "feature_ranges": feature_ranges(X_train),
        "baseline": baseline_profile(X_train),
        "ood": build_ood_detector(model, X_train, y_train),
    }
    joblib.dump(artifact, "employee_attrition_model.pkl")
    print("Saved model + metadata to employee_attrition_model.pkl")


if __name__ == "__main__":
    main()
