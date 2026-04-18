"""Model loading and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class ModelLoadError(RuntimeError):
    """Raised when the configured model cannot be loaded."""


def load_model(model_path: str | Path) -> Any:
    """Load a trusted pickle/joblib sklearn object from disk.

    Pickle files can execute arbitrary code when loaded. Only use model files
    created by your own training pipeline or another trusted source.
    """

    path = Path(model_path)
    if path.suffix not in {".pkl", ".pickle", ".joblib"}:
        raise ModelLoadError("El modelo debe tener extension .pkl, .pickle o .joblib.")
    if not path.exists():
        raise ModelLoadError(f"No se encontro el modelo en: {path}")
    if not path.is_file():
        raise ModelLoadError(f"La ruta del modelo no es un archivo: {path}")

    try:
        model = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"No se pudo cargar el modelo: {exc}") from exc

    if not hasattr(model, "predict"):
        raise ModelLoadError("El objeto cargado no expone un metodo predict().")

    return model

