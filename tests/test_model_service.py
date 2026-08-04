import numpy as np
import pytest

import model_service


class _FakeModel:
    """Deterministic stand-in so predict_one can be tested without a real fit."""

    def predict_proba(self, row):
        return np.array([[0.65, 0.35]])


@pytest.fixture
def bundle():
    return {
        "model": _FakeModel(),
        "threshold": 0.5,
        "features": ["satisfaction_level", "Department"],
    }


def test_predict_one_returns_probability_flag_and_latency(bundle):
    result = model_service.predict_one(bundle, {"satisfaction_level": 0.4, "Department": "sales"})

    assert result["probability_leave"] == pytest.approx(0.35)
    assert result["will_leave"] is False  # 0.35 < threshold 0.5
    assert result["latency_ms"] >= 0


def test_predict_one_flags_leave_when_above_threshold():
    bundle = {
        "model": _FakeModel(),
        "threshold": 0.3,
        "features": ["satisfaction_level"],
    }
    result = model_service.predict_one(bundle, {"satisfaction_level": 0.4})

    assert result["will_leave"] is True  # 0.35 >= threshold 0.3


def test_predict_one_reindexes_to_the_bundles_feature_order(bundle):
    # Extra/misordered keys in the input shouldn't matter -- reindex to
    # bundle["features"] is what keeps prediction order stable.
    result = model_service.predict_one(
        bundle, {"Department": "sales", "satisfaction_level": 0.4, "extra_unused_field": 999}
    )

    assert result["probability_leave"] == pytest.approx(0.35)


# --------------------------------------------------------------------------
# explain
# --------------------------------------------------------------------------


class _SatisfactionDrivenModel:
    """Probability depends only on satisfaction, so contributions are known."""

    def __init__(self):
        self.named_steps = {"prep": _FakePreprocessor()}

    def predict_proba(self, frame):
        leave = (10 - frame["satisfaction_level"].to_numpy()) / 10
        return np.column_stack([1 - leave, leave])


def _explainable_bundle():
    return {
        "model": _SatisfactionDrivenModel(),
        "threshold": 0.5,
        "features": ["satisfaction_level", "average_monthly_hours"],
        "baseline": {"satisfaction_level": 5.0, "average_monthly_hours": 200.0},
    }


def test_explain_attributes_the_move_to_the_field_that_caused_it():
    # satisfaction 1 -> 0.9 leave probability; baseline 5 -> 0.5. So
    # satisfaction contributes +40 points and hours nothing.
    drivers = dict(
        model_service.explain(
            _explainable_bundle(), {"satisfaction_level": 1.0, "average_monthly_hours": 200.0}
        )
    )

    assert drivers["satisfaction_level"] == pytest.approx(40.0)
    assert drivers["average_monthly_hours"] == pytest.approx(0.0)


def test_explain_signs_a_risk_lowering_field_negative():
    drivers = dict(
        model_service.explain(
            _explainable_bundle(), {"satisfaction_level": 9.0, "average_monthly_hours": 200.0}
        )
    )

    assert drivers["satisfaction_level"] == pytest.approx(-40.0)


def test_explain_orders_by_size_of_effect_and_truncates():
    drivers = model_service.explain(
        _explainable_bundle(),
        {"satisfaction_level": 1.0, "average_monthly_hours": 200.0},
        top_n=1,
    )

    assert len(drivers) == 1
    assert drivers[0][0] == "satisfaction_level"


def test_explain_returns_nothing_without_a_baseline():
    bundle = _explainable_bundle()
    del bundle["baseline"]

    assert model_service.explain(bundle, {"satisfaction_level": 1.0}) == []


# --------------------------------------------------------------------------
# same_profile
# --------------------------------------------------------------------------

PROFILE_FEATURES = ["satisfaction_level", "projects_worked_on", "Department"]


def test_same_profile_matches_identical_inputs():
    row = {"satisfaction_level": 4.0, "projects_worked_on": 3, "Department": "sales"}
    assert model_service.same_profile(row, dict(row), PROFILE_FEATURES) is True


def test_same_profile_ignores_float_int_representation():
    """Widgets hand back 3 where the dataframe held 3.0; that's the same
    employee, and the recorded outcome should still apply."""
    from_widget = {"satisfaction_level": 4.0, "projects_worked_on": 3, "Department": "sales"}
    from_frame = {"satisfaction_level": 4.0, "projects_worked_on": 3.0, "Department": "sales"}
    assert model_service.same_profile(from_widget, from_frame, PROFILE_FEATURES) is True


def test_same_profile_rejects_a_nudged_slider():
    """The case that matters: nudge one input and the known outcome belongs
    to a different person, so it must stop being reported."""
    original = {"satisfaction_level": 4.0, "projects_worked_on": 3, "Department": "sales"}
    nudged = {**original, "satisfaction_level": 4.1}
    assert model_service.same_profile(original, nudged, PROFILE_FEATURES) is False


def test_same_profile_rejects_a_changed_category():
    original = {"satisfaction_level": 4.0, "projects_worked_on": 3, "Department": "sales"}
    moved = {**original, "Department": "technical"}
    assert model_service.same_profile(original, moved, PROFILE_FEATURES) is False


def test_same_profile_rejects_an_empty_comparison():
    row = {"satisfaction_level": 4.0, "projects_worked_on": 3, "Department": "sales"}
    assert model_service.same_profile(row, {}, PROFILE_FEATURES) is False


# --------------------------------------------------------------------------
# risk_band
# --------------------------------------------------------------------------

BASE_RATE = 0.166  # company-wide attrition rate after deduplication
DECISION_THRESHOLD = 0.384


def band(probability):
    return model_service.risk_band(probability, DECISION_THRESHOLD, BASE_RATE)


def test_probability_above_the_decision_threshold_is_high_risk():
    assert band(0.9) == "high"
    assert band(DECISION_THRESHOLD) == "high"


def test_probability_above_the_company_average_is_elevated_not_low():
    """The bug this fixes: 25% is 1.5x the company-wide rate, but a bare
    below-threshold test rendered it as a reassuring "likely to stay"."""
    assert band(0.25) == "elevated"
    assert band(0.30) == "elevated"


def test_probability_below_the_company_average_is_low_risk():
    assert band(0.02) == "low"
    assert band(0.15) == "low"


def test_band_boundary_sits_exactly_at_the_company_average():
    assert band(BASE_RATE - 0.001) == "low"
    assert band(BASE_RATE) == "elevated"


def test_predict_one_reports_the_band_and_base_rate(bundle):
    bundle["base_rate"] = BASE_RATE
    result = model_service.predict_one(bundle, {"satisfaction_level": 0.4, "Department": "sales"})

    # Fake model returns 0.35, threshold 0.5 -> below the flag threshold but
    # more than double the company average, so "elevated", not a green "low".
    assert result["risk_band"] == "elevated"
    assert result["base_rate"] == BASE_RATE


# --------------------------------------------------------------------------
# Out-of-distribution guard
# --------------------------------------------------------------------------


class _FakeIndex:
    """Returns a fixed nearest-neighbour distance, standing in for FAISS/sklearn."""

    def __init__(self, distance):
        self.distance = distance

    def kneighbors(self, embedded):
        return np.array([[self.distance]]), np.array([[0]])


class _FakePreprocessor:
    def transform(self, row):
        return np.zeros((1, 3))


class _FakeModelWithPrep(_FakeModel):
    def __init__(self):
        self.named_steps = {"prep": _FakePreprocessor()}


def _bundle_with_ood(distance, threshold=2.0):
    return {
        "model": _FakeModelWithPrep(),
        "threshold": 0.5,
        "features": ["satisfaction_level"],
        "ood": {"index": _FakeIndex(distance), "threshold": threshold},
    }


def test_profile_close_to_training_data_is_not_flagged():
    result = model_service.predict_one(_bundle_with_ood(distance=0.5), {"satisfaction_level": 5})

    assert result["out_of_distribution"] is False
    assert result["novelty_score"] == pytest.approx(0.25)


def test_profile_far_from_training_data_is_flagged():
    """The case worth caveating: every value in range, but so unlike any
    real employee that the forest's number is mostly extrapolation."""
    result = model_service.predict_one(_bundle_with_ood(distance=6.0), {"satisfaction_level": 5})

    assert result["out_of_distribution"] is True
    assert result["novelty_score"] == pytest.approx(3.0)


def test_mildly_novel_profile_is_not_flagged():
    """Most in-range profiles a user builds by hand land past 1.0x -- 60% of
    randomly sampled ones do. Caveating those would make the note constant
    noise, so the bar is CAUTION_NOVELTY, not 1.0."""
    result = model_service.predict_one(_bundle_with_ood(distance=3.0), {"satisfaction_level": 5})

    assert result["novelty_score"] == pytest.approx(1.5)
    assert result["out_of_distribution"] is False


def test_caution_boundary_is_the_configured_multiple():
    at = model_service.CAUTION_NOVELTY * 2.0  # bundle threshold is 2.0
    assert (
        model_service.predict_one(_bundle_with_ood(distance=at), {"satisfaction_level": 5})[
            "out_of_distribution"
        ]
        is False
    )
    assert (
        model_service.predict_one(_bundle_with_ood(distance=at + 0.01), {"satisfaction_level": 5})[
            "out_of_distribution"
        ]
        is True
    )


def test_bundle_without_ood_detector_never_flags(bundle):
    result = model_service.predict_one(bundle, {"satisfaction_level": 0.4, "Department": "sales"})

    assert result["out_of_distribution"] is False
    assert result["novelty_score"] == 0.0


# --------------------------------------------------------------------------
# load_or_train_model
# --------------------------------------------------------------------------


def test_load_or_train_model_loads_a_complete_artifact(tmp_path):
    import joblib

    artifact = {
        "model": _FakeModel(),
        "threshold": 0.4,
        "features": ["a", "b"],
        "base_rate": 0.24,
        "feature_ranges": {"a": (0.0, 1.0)},
        "baseline": {"a": 0.5},
        "ood": {"index": _FakeIndex(0.1), "threshold": 1.0},
    }
    path = tmp_path / "model.pkl"
    joblib.dump(artifact, path)

    loaded = model_service.load_or_train_model(str(path))

    assert loaded["threshold"] == 0.4
    assert loaded["features"] == ["a", "b"]


def test_incomplete_artifact_is_retrained_rather_than_served(tmp_path, monkeypatch):
    """An artifact predating the OOD guard would otherwise be served with
    the guard silently missing, so every request would look in-distribution.
    It must be retrained instead of half-used."""
    import joblib
    import pandas as pd

    import train

    stale = {"model": _FakeModel(), "threshold": 0.4, "features": ["stale_feature"]}
    path = tmp_path / "model.pkl"
    joblib.dump(stale, path)

    frame = pd.DataFrame({"fresh_feature": [1, 2, 3, 4]})
    labels = pd.Series([0, 0, 0, 1])
    monkeypatch.setattr(train, "load_data", lambda *a, **kw: frame)
    monkeypatch.setattr(
        train, "split_data", lambda *a, **kw: (frame, frame, frame, labels, labels, labels)
    )
    monkeypatch.setattr(train, "train_model", lambda *a, **kw: (_FakeModel(), {}, 0.9))
    monkeypatch.setattr(train, "tune_threshold", lambda *a, **kw: 0.42)
    monkeypatch.setattr(train, "feature_ranges", lambda *a, **kw: {"fresh_feature": (1.0, 4.0)})
    monkeypatch.setattr(train, "baseline_profile", lambda *a, **kw: {"fresh_feature": 2.0})
    monkeypatch.setattr(
        train, "build_ood_detector", lambda *a, **kw: {"index": _FakeIndex(0.1), "threshold": 1.0}
    )

    bundle = model_service.load_or_train_model(str(path))

    assert bundle["features"] == ["fresh_feature"], "stale artifact was served instead of retrained"
    assert bundle["threshold"] == 0.42
    assert model_service.REQUIRED_KEYS <= set(bundle)


# --------------------------------------------------------------------------
# similar_employees
# --------------------------------------------------------------------------


class _NeighbourIndex:
    """Returns fixed neighbour positions, standing in for the fitted index."""

    def __init__(self, positions):
        self.positions = positions

    def kneighbors(self, embedded, n_neighbors=None):
        chosen = self.positions[: n_neighbors or len(self.positions)]
        return np.array([[0.1] * len(chosen)]), np.array([chosen])


def _bundle_with_outcomes(positions, outcomes):
    return {
        "model": _FakeModelWithPrep(),
        "threshold": 0.5,
        "features": ["satisfaction_level"],
        "ood": {
            "index": _NeighbourIndex(positions),
            "threshold": 2.0,
            "outcomes": np.array(outcomes),
        },
    }


def test_similar_employees_counts_how_many_of_them_left():
    bundle = _bundle_with_outcomes([0, 1, 2, 3], [1, 1, 0, 0])

    assert model_service.similar_employees(bundle, {"satisfaction_level": 2.0}, k=4) == (2, 4)


def test_similar_employees_reports_none_left_when_all_stayed():
    bundle = _bundle_with_outcomes([0, 1, 2], [0, 0, 0])

    assert model_service.similar_employees(bundle, {"satisfaction_level": 2.0}, k=3) == (0, 3)


def test_similar_employees_never_asks_for_more_neighbours_than_exist():
    bundle = _bundle_with_outcomes([0, 1], [1, 0])

    left, total = model_service.similar_employees(bundle, {"satisfaction_level": 2.0}, k=50)
    assert (left, total) == (1, 2)


def test_similar_employees_is_absent_for_an_artifact_without_outcomes():
    """Older artifacts stored no labels; the panel hides rather than lying."""
    bundle = _bundle_with_outcomes([0, 1], [1, 0])
    bundle["ood"]["outcomes"] = None

    assert model_service.similar_employees(bundle, {"satisfaction_level": 2.0}) is None
