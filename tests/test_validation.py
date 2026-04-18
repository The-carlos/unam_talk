import pandas as pd
import pytest

from src.validation import ValidationError, read_csv_file, select_model_columns, validate_columns


def test_validate_columns_reports_missing_columns():
    df = pd.DataFrame({"columna_1": [1], "columna_2": [2]})

    result = validate_columns(df, ["columna_1", "columna_2", "columna_3"])

    assert result.is_valid is False
    assert result.missing_columns == ["columna_3"]


def test_select_model_columns_preserves_expected_order():
    df = pd.DataFrame({"b": [1], "a": [2], "c": [3]})

    selected = select_model_columns(df, ["a", "b"])

    assert selected.columns.tolist() == ["a", "b"]


def test_read_csv_file_rejects_empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")

    with pytest.raises(ValidationError):
        read_csv_file(path)

