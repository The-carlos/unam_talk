"""Prediction orchestration independent from the Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from src.validation import ValidationError, select_model_columns, validate_columns
except ModuleNotFoundError:
    from validation import ValidationError, select_model_columns, validate_columns


@dataclass(frozen=True)
class PredictionSummary:
    rows_processed: int
    columns_received: int
    prediction_distribution: dict[str, int]
    messages: list[str]


@dataclass(frozen=True)
class RankedPrediction:
    label: str
    probability: float


def _normalize_categorical_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Cast boolean categorical values to sklearn-friendly string labels."""

    result = df.copy()
    for column in result.columns:
        series = result[column]

        if pd.api.types.is_bool_dtype(series):
            result[column] = series.map(lambda value: "True" if value else "False")
            continue

        if pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            if not non_null.empty and non_null.map(lambda value: isinstance(value, bool)).any():
                result[column] = series.map(
                    lambda value: "True" if value is True else ("False" if value is False else value)
                )

    return result


def generate_predictions(
    model,
    input_df: pd.DataFrame,
    expected_columns: list[str],
    prediction_column: str = "prediction",
) -> tuple[pd.DataFrame, PredictionSummary]:
    """Validate input data, run predictions, and append them to the original data."""

    validation = validate_columns(input_df, expected_columns)
    if not validation.is_valid:
        raise ValidationError("; ".join(validation.messages))

    model_input = select_model_columns(input_df, expected_columns)
    model_input = _normalize_categorical_booleans(model_input)
    predictions = model.predict(model_input)

    output_df = input_df.copy()
    output_df[prediction_column] = predictions

    distribution = (
        pd.Series(predictions, name=prediction_column)
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    summary = PredictionSummary(
        rows_processed=len(output_df),
        columns_received=len(input_df.columns),
        prediction_distribution={str(key): int(value) for key, value in distribution.items()},
        messages=validation.messages,
    )
    return output_df, summary


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a dataframe to CSV bytes for browser download."""

    return df.to_csv(index=False).encode("utf-8")


def generate_top_k_predictions(
    model,
    input_df: pd.DataFrame,
    expected_columns: list[str],
    top_k: int = 3,
) -> list[RankedPrediction]:
    """Return ranked predictions (label + probability) for a single/batch dataframe."""

    validation = validate_columns(input_df, expected_columns)
    if not validation.is_valid:
        raise ValidationError("; ".join(validation.messages))

    if not hasattr(model, "predict_proba"):
        raise ValidationError("El modelo no soporta probabilidades (predict_proba).")

    model_input = select_model_columns(input_df, expected_columns)
    model_input = _normalize_categorical_booleans(model_input)

    probabilities = model.predict_proba(model_input)
    if len(probabilities) == 0:
        raise ValidationError("No se pudieron calcular probabilidades para el input.")

    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValidationError("No se pudieron leer las clases del modelo.")

    ranked = (
        pd.DataFrame({"label": classes, "probability": probabilities[0]})
        .sort_values("probability", ascending=False)
        .head(top_k)
    )

    return [
        RankedPrediction(label=str(row.label), probability=float(row.probability))
        for row in ranked.itertuples(index=False)
    ]
