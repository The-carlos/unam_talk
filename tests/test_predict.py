import pandas as pd
import pytest

from src.predict import (
    _normalize_categorical_booleans,
    dataframe_to_csv_bytes,
    generate_predictions,
    generate_top_k_predictions,
)
from src.validation import ValidationError


class DummyModel:
    def predict(self, X):
        return ["yes" if value > 0 else "no" for value in X["columna_1"]]

    def predict_proba(self, X):
        return [[0.2, 0.8] for _ in range(len(X))]

    @property
    def classes_(self):
        return ["no", "yes"]


def test_generate_predictions_appends_prediction_column():
    df = pd.DataFrame({"columna_1": [1, -1], "columna_2": ["a", "b"]})

    output_df, summary = generate_predictions(
        model=DummyModel(),
        input_df=df,
        expected_columns=["columna_1", "columna_2"],
        prediction_column="prediction",
    )

    assert output_df["prediction"].tolist() == ["yes", "no"]
    assert summary.rows_processed == 2
    assert summary.prediction_distribution == {"no": 1, "yes": 1}


def test_generate_predictions_rejects_missing_columns():
    df = pd.DataFrame({"columna_1": [1]})

    with pytest.raises(ValidationError):
        generate_predictions(DummyModel(), df, ["columna_1", "columna_2"])


def test_dataframe_to_csv_bytes_returns_utf8_csv():
    df = pd.DataFrame({"a": [1]})

    assert dataframe_to_csv_bytes(df) == b"a\n1\n"


def test_normalize_categorical_booleans_casts_bool_values_to_strings():
    df = pd.DataFrame(
        {
            "Legendary": [True, False],
            "Type 1": ["Fire", "Water"],
            "Total": [500, 530],
        }
    )

    normalized = _normalize_categorical_booleans(df)

    assert normalized["Legendary"].tolist() == ["True", "False"]


def test_generate_top_k_predictions_returns_ranked_results():
    df = pd.DataFrame({"columna_1": [1], "columna_2": ["a"]})

    ranked = generate_top_k_predictions(
        model=DummyModel(),
        input_df=df,
        expected_columns=["columna_1", "columna_2"],
        top_k=2,
    )

    assert ranked[0].label == "yes"
    assert ranked[0].probability == 0.8
    assert ranked[1].label == "no"
