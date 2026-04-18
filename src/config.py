"""Central configuration for local and container execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    model_path: Path
    expected_columns: list[str]
    prediction_column: str


def get_settings() -> Settings:
    """Load runtime settings from environment variables."""

    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    return Settings(
        model_path=model_path,
        expected_columns=_split_csv(os.getenv("EXPECTED_COLUMNS")),
        prediction_column=os.getenv("PREDICTION_COLUMN", "prediction"),
    )

