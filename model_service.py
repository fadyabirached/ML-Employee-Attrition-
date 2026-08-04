"""Shared model loading + prediction logic for app.py and api.py.

Both interfaces need the same things: a trained model bundle, a way to
turn one raw feature dict into a probability, and a way to tell when a
request doesn't resemble anyone the model was trained on. Keeping that
logic here means the Streamlit UI and the FastAPI service can't drift
apart -- there's exactly one place that knows how a prediction is made.
"""
from __future__ import annotations

import os
import time
from typing import Any

import joblib
import pandas as pd

MODEL_PATH = "employee_attrition_model.pkl"

# The hyperparameters GridSearchCV selected (see README "Modeling & tuning").
# Used as a fast, no-search fallback so a deployment without the committed
# artifact only has to fit one model, not repeat the full grid search on
# every cold start.
BEST_PARAMS = {"learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 20}

# An artifact missing any of these predates them; retrain rather than serve
# a bundle whose out-of-distribution guard is silently absent.
REQUIRED_KEYS = {
    "model",
    "threshold",
    "features",
    "base_rate",
    "feature_ranges",
    "baseline",
    "ood",
}

# How novel a profile must be before its estimate is worth caveating.
#
# Not 1.0: the data occupies a thin manifold inside its own bounding box, so
# 60% of randomly sampled in-range profiles sit past 1.0 and a caveat that
# common is noise nobody reads. Measured across 300 random in-range profiles:
# 60% exceed 1.0x, 18% exceed 1.5x, 2% exceed 2.0x. At 2.0x the note stays
# rare enough to mean something when it does appear.
CAUTION_NOVELTY = 2.0


def load_or_train_model(path: str = MODEL_PATH) -> dict:
    """Load the trained model bundle, training a fresh one if it's missing.

    Returns a dict shaped like {"model", "threshold", "features",
    "feature_ranges", "ood"} -- the same structure train.py's main() saves
    via joblib.
    """
    if os.path.exists(path):
        bundle = joblib.load(path)
        if REQUIRED_KEYS <= set(bundle):
            return bundle

    import train

    df = train.load_data()
    X_train, X_val, _X_test, y_train, y_val, _y_test = train.split_data(df)

    fixed_grid = {
        "clf__learning_rate": [BEST_PARAMS["learning_rate"]],
        "clf__max_leaf_nodes": [BEST_PARAMS["max_leaf_nodes"]],
        "clf__min_samples_leaf": [BEST_PARAMS["min_samples_leaf"]],
    }
    model, _best_params, _cv_f1 = train.train_model(
        X_train, y_train, param_grid=fixed_grid, cv_splits=2
    )
    threshold = train.tune_threshold(model, X_val, y_val)

    return {
        "model": model,
        "threshold": threshold,
        "features": list(X_train.columns),
        "base_rate": float(y_train.mean()),
        "feature_ranges": train.feature_ranges(X_train),
        "baseline": train.baseline_profile(X_train),
        "ood": train.build_ood_detector(model, X_train, y_train),
    }


def same_profile(left: dict, right: dict, features: list) -> bool:
    """Whether two feature dicts describe the same employee.

    Used to decide if a known outcome still belongs to what's on screen:
    load a real employee, nudge a slider, and the recorded answer no
    longer applies to the profile being scored. Numerics are compared at
    one decimal place, which is the resolution the inputs offer.
    """
    def canonical(values):
        out = []
        for name in features:
            value = values.get(name)
            out.append(round(value, 1) if isinstance(value, (int, float)) else value)
        return out

    return canonical(left) == canonical(right)


def novelty_score(bundle: dict, row: pd.DataFrame) -> float:
    """How far a request sits from the training data, relative to the limit.

    1.0 means it sits exactly as far from the data as the most isolated
    training row sits from its own neighbour. Anything above that is a
    profile the model has no comparable example for, so its probability is
    extrapolation rather than evidence. See train.build_ood_detector.
    """
    detector = bundle.get("ood")
    if detector is None:
        return 0.0

    embedded = bundle["model"].named_steps["prep"].transform(row)
    distance, _ = detector["index"].kneighbors(embedded)
    return float(distance[0][0] / detector["threshold"])


def similar_employees(bundle: dict, features: dict[str, Any], k: int = 15):
    """What became of the real employees this request most resembles.

    Returns ``(left, total)`` or None when the artifact predates the stored
    outcomes. This is the only claim on the page that does not depend on
    trusting the model: whatever the probability says, these are actual
    people from the training data and that is what actually happened to
    them.
    """
    detector = bundle.get("ood") or {}
    outcomes = detector.get("outcomes")
    if outcomes is None:
        return None

    row = pd.DataFrame([features]).reindex(columns=bundle["features"])
    embedded = bundle["model"].named_steps["prep"].transform(row)
    k = min(k, len(outcomes))
    _distances, positions = detector["index"].kneighbors(embedded, n_neighbors=k)
    neighbours = outcomes[positions[0]]
    return int(neighbours.sum()), int(k)


def explain(bundle: dict, features: dict[str, Any], top_n: int = 4) -> list:
    """Which inputs moved this prediction, and by how much.

    Swaps one field at a time for an ordinary employee's value and records
    how far the probability moves. Positive means that field pushed the
    risk up relative to a typical employee, negative that it pulled it
    down. Returned in percentage points, largest effect first.

    Attrition in this data comes from combinations rather than any single
    field, so a result read on its own regularly looks arbitrary -- an
    unhappy employee can be low risk because of tenure and workload. This
    is what turns that from arbitrary into legible.

    All the swaps go through one predict_proba call rather than one per
    field, which keeps it to a single pass over the forest.
    """
    baseline = bundle.get("baseline")
    if not baseline:
        return []

    columns = bundle["features"]
    swappable = [name for name in columns if name in baseline]
    rows = [features] + [{**features, name: baseline[name]} for name in swappable]

    probabilities = bundle["model"].predict_proba(
        pd.DataFrame(rows).reindex(columns=columns)
    )[:, 1]

    actual = probabilities[0]
    effects = [
        (name, float((actual - probabilities[index + 1]) * 100))
        for index, name in enumerate(swappable)
    ]
    effects.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return effects[:top_n]


def risk_band(probability: float, decision_threshold: float, base_rate: float) -> str:
    """Bucket a probability against two reference points a reader can check.

    ``base_rate`` is the company-wide attrition rate; ``decision_threshold``
    is the tuned cutoff for flagging someone. Reporting "below the
    threshold" alone lumps 5% and 30% together as reassuring, even though
    30% is nearly twice the company average -- so anything at or above the
    base rate reads as elevated rather than safe.
    """
    if probability >= decision_threshold:
        return "high"
    if probability >= base_rate:
        return "elevated"
    return "low"


def predict_one(bundle: dict, features: dict[str, Any]) -> dict:
    """Score a single employee record.

    ``out_of_distribution`` is the one to check first: when it's true the
    probability is not a trustworthy answer, just what the forest happened
    to output for a profile it has never seen. ``risk_band`` is the
    honest reading of the probability itself -- see :func:`risk_band`.
    """
    row = pd.DataFrame([features]).reindex(columns=bundle["features"])

    model = bundle["model"]
    _ = model.predict_proba(row)  # warm-up, keeps the timed call honest

    start = time.perf_counter()
    prob_leave = float(model.predict_proba(row)[:, 1][0])
    latency_ms = (time.perf_counter() - start) * 1000

    novelty = novelty_score(bundle, row)
    base_rate = bundle.get("base_rate", 0.0)

    return {
        "probability_leave": prob_leave,
        "will_leave": bool(prob_leave >= bundle["threshold"]),
        "risk_band": risk_band(prob_leave, bundle["threshold"], base_rate),
        "base_rate": base_rate,
        "out_of_distribution": novelty > CAUTION_NOVELTY,
        "novelty_score": novelty,
        "latency_ms": latency_ms,
    }
