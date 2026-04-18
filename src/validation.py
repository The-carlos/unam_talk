"""Input validation utilities for uploaded prediction files."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import BinaryIO

import pandas as pd


class ValidationError(ValueError):
    """Raised when an input file cannot be used for prediction."""


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    messages: list[str]
    missing_columns: list[str]
    extra_columns: list[str]


def read_csv_file(file: str | BytesIO | StringIO | BinaryIO) -> pd.DataFrame:
    """Read a CSV file into a dataframe with friendly validation errors."""

    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError as exc:
        raise ValidationError("El archivo CSV esta vacio.") from exc
    except UnicodeDecodeError as exc:
        raise ValidationError("No se pudo leer el archivo. Verifica que use codificacion UTF-8.") from exc
    except Exception as exc:
        raise ValidationError(f"No se pudo leer el CSV: {exc}") from exc

    if df.empty:
        raise ValidationError("El archivo CSV no contiene filas para procesar.")
    if len(df.columns) == 0:
        raise ValidationError("El archivo CSV no contiene columnas.")

    return df


def validate_columns(df: pd.DataFrame, expected_columns: list[str]) -> ValidationResult:
    """Validate that the dataframe contains the columns required by the model."""

    if not expected_columns:
        return ValidationResult(
            is_valid=True,
            messages=["No se configuraron columnas esperadas; se usaran las columnas del CSV."],
            missing_columns=[],
            extra_columns=[],
        )

    observed = set(df.columns)
    expected = set(expected_columns)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    messages: list[str] = []
    if missing:
        messages.append("Faltan columnas requeridas: " + ", ".join(missing))
    else:
        messages.append("El CSV contiene todas las columnas requeridas.")

    if extra:
        messages.append("Columnas adicionales detectadas: " + ", ".join(extra))

    return ValidationResult(
        is_valid=not missing,
        messages=messages,
        missing_columns=missing,
        extra_columns=extra,
    )


def select_model_columns(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Return dataframe columns in the order expected by the trained pipeline."""

    if not expected_columns:
        return df
    return df.loc[:, expected_columns]

