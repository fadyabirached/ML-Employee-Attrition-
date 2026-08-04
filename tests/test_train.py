import numpy as np
import pandas as pd
import pytest

import train


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv("dataset.csv")


def test_load_data_drops_exact_duplicates(raw_df):
    assert raw_df.duplicated().sum() > 0, "fixture assumption: raw dataset has duplicate rows"

    df = train.load_data()

    assert df.duplicated().sum() == 0
    assert len(df) < len(raw_df)


def test_split_data_is_stratified_and_disjoint(raw_df):
    df = train.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = train.split_data(df)

    assert len(X_train) + len(X_val) + len(X_test) == len(df)

    train_idx, val_idx, test_idx = set(X_train.index), set(X_val.index), set(X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)

    overall_rate = df[train.TARGET].mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - overall_rate) < 0.03


def test_build_preprocessor_output_shape():
    df = train.load_data()
    X = df.drop(train.TARGET, axis=1)

    preprocessor = train.build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    n_numeric = len(train.NUMERIC_FEATURES)
    n_ordinal = len(train.ORDINAL_FEATURES)
    n_categorical_dummies = sum(X[c].nunique() - 1 for c in train.CATEGORICAL_FEATURES)
    n_binary = len(train.BINARY_FEATURES)
    assert transformed.shape == (
        len(X),
        n_numeric + n_ordinal + n_categorical_dummies + n_binary,
    )


def test_binary_features_actually_reach_the_model():
    """ColumnTransformer's remainder defaults to "drop", which silently
    discarded Work_accident and promotion_last_5years -- both ~3-4x
    predictive -- and made their UI toggles inert."""
    df = train.load_data()
    X = df.drop(train.TARGET, axis=1)

    preprocessor = train.build_preprocessor()
    preprocessor.fit(X)

    passed_through = set()
    for name, _transformer, columns in preprocessor.transformers_:
        if name != "remainder":
            passed_through.update(columns)

    assert set(train.BINARY_FEATURES) <= passed_through


def test_pipeline_trains_and_predicts_on_a_small_grid():
    """Fast end-to-end smoke test with a trivial grid, not the full search."""
    df = train.load_data().sample(n=500, random_state=0).reset_index(drop=True)
    X_train, X_val, X_test, y_train, y_val, y_test = train.split_data(df)

    tiny_grid = {"clf__max_iter": [20], "clf__max_leaf_nodes": [8], "clf__min_samples_leaf": [5]}
    model, _best_params, cv_f1 = train.train_model(X_train, y_train, param_grid=tiny_grid, cv_splits=2)

    assert 0.0 <= cv_f1 <= 1.0

    threshold = train.tune_threshold(model, X_val, y_val)
    assert 0.0 <= threshold <= 1.0

    probs = model.predict_proba(X_test)[:, 1]
    assert probs.shape == (len(X_test),)
    assert np.all((probs >= 0) & (probs <= 1))

    report = train.evaluate(model, threshold, X_test, y_test)
    assert "Stayed" in report and "Left" in report


# --------------------------------------------------------------------------
# Monotonic constraints
# --------------------------------------------------------------------------


def test_constraint_vector_lines_up_with_transformed_column_order():
    """The constraint list sklearn takes is positional, so a mismatch would
    quietly constrain the wrong column rather than fail."""
    df = train.load_data()
    preprocessor = train.build_preprocessor().fit(df.drop(train.TARGET, axis=1))

    names = list(preprocessor.get_feature_names_out())
    vector = train.monotonic_constraint_vector(preprocessor)

    assert len(vector) == len(names)
    for name, constraint in zip(names, vector):
        assert constraint == train.MONOTONIC_CONSTRAINTS.get(name, 0)
    # Every constraint we declared must have matched a real column.
    assert sum(1 for c in vector if c != 0) == len(train.MONOTONIC_CONSTRAINTS)


def test_salary_is_ordinal_so_low_medium_high_stays_ordered():
    """As separate one-hot dummies nothing forces low >= medium; ordinal
    encoding is what lets a single constraint cover the whole ladder."""
    df = train.load_data()
    preprocessor = train.build_preprocessor().fit(df.drop(train.TARGET, axis=1))

    assert "ord__salary" in list(preprocessor.get_feature_names_out())

    encoded = preprocessor.named_transformers_["ord"].transform(
        pd.DataFrame({"salary": train.SALARY_ORDER})
    )
    assert list(encoded.ravel()) == [0.0, 1.0, 2.0]


def test_a_more_satisfied_employee_is_never_scored_as_higher_risk():
    """The behaviour the constraints exist for. Without them the model can
    report a more satisfied employee as more likely to leave, which is
    indefensible to a user whatever the accuracy is."""
    df = train.load_data().sample(n=3000, random_state=0).reset_index(drop=True)
    X_train, _X_val, _X_test, y_train, _y_val, _y_test = train.split_data(df)

    tiny_grid = {"clf__max_iter": [40], "clf__max_leaf_nodes": [16], "clf__min_samples_leaf": [10]}
    model, _params, _cv = train.train_model(X_train, y_train, param_grid=tiny_grid, cv_splits=2)

    base = X_train.iloc[0].to_dict()
    sweep = pd.DataFrame(
        [{**base, "satisfaction_level": level} for level in np.arange(0.9, 10.01, 0.5)]
    )
    probabilities = model.predict_proba(sweep)[:, 1]

    assert np.all(np.diff(probabilities) <= 1e-9), "risk rose as satisfaction rose"
