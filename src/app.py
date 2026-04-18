"""Streamlit frontend for batch CSV predictions and interactive quiz."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

try:
    from src.config import get_settings
    from src.features import normalize_column_names
    from src.model_io import ModelLoadError, load_model
    from src.predict import dataframe_to_csv_bytes, generate_predictions, generate_top_k_predictions
    from src.validation import ValidationError, read_csv_file
except ModuleNotFoundError:
    # Fallback when Streamlit executes from the src/ directory context.
    from config import get_settings
    from features import normalize_column_names
    from model_io import ModelLoadError, load_model
    from predict import dataframe_to_csv_bytes, generate_predictions, generate_top_k_predictions
    from validation import ValidationError, read_csv_file


st.set_page_config(page_title="Pokemon ML Production Demo", layout="wide")


NUMERIC_FEATURE_CONFIG = {
    "Total": {"min_value": 180, "max_value": 780, "value": 500, "step": 1},
    "HP": {"min_value": 1, "max_value": 255, "value": 80, "step": 1},
    "Attack": {"min_value": 1, "max_value": 255, "value": 85, "step": 1},
    "Defense": {"min_value": 1, "max_value": 255, "value": 75, "step": 1},
    "Sp. Atk": {"min_value": 1, "max_value": 255, "value": 85, "step": 1},
    "Sp. Def": {"min_value": 1, "max_value": 255, "value": 75, "step": 1},
    "Speed": {"min_value": 1, "max_value": 255, "value": 80, "step": 1},
    "Generation": {"min_value": 1, "max_value": 9, "value": 1, "step": 1},
}


@st.cache_resource(show_spinner=False)
def cached_model(model_path: str):
    return load_model(model_path)


def _extract_categorical_options(model: Any) -> dict[str, list[str]]:
    """Read categorical options from the fitted sklearn preprocessor, if available."""

    options: dict[str, list[str]] = {}

    try:
        preprocessor = model.named_steps["preprocessor"]
        cat_columns: list[str] | None = None

        for name, _transformer, columns in preprocessor.transformers_:
            if name == "cat":
                cat_columns = [str(column) for column in columns]
                break

        if not cat_columns:
            return options

        cat_pipeline = preprocessor.named_transformers_["cat"]
        encoder = cat_pipeline.named_steps["onehot"]

        for column, values in zip(cat_columns, encoder.categories_):
            options[column] = sorted({str(value) for value in values})
    except Exception:
        return {}

    return options


def _resolve_expected_columns(settings, model: Any) -> list[str]:
    if settings.expected_columns:
        return settings.expected_columns

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        return []

    return [str(feature) for feature in model_features]


def _render_batch_tab(model: Any, settings, expected_columns: list[str]) -> None:
    st.subheader("Seccion 1: Prediccion batch por CSV")
    st.write("Carga un CSV, valida columnas y descarga un archivo con predicciones.")

    uploaded_file = st.file_uploader("Archivo CSV", type=["csv"], key="batch_uploader")
    if uploaded_file is None:
        st.info("Sube un archivo .csv para generar predicciones.")
        return

    try:
        input_df = normalize_column_names(read_csv_file(uploaded_file))
        output_df, summary = generate_predictions(
            model=model,
            input_df=input_df,
            expected_columns=expected_columns,
            prediction_column=settings.prediction_column,
        )
    except ValidationError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"No se pudieron generar predicciones: {exc}")
        return

    st.success("Predicciones generadas correctamente.")

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Filas procesadas", summary.rows_processed)
    metric_b.metric("Columnas recibidas", summary.columns_received)
    metric_c.metric("Clases predichas", len(summary.prediction_distribution))

    st.subheader("Mensajes de validacion")
    for message in summary.messages:
        st.write(f"- {message}")

    st.subheader("Distribucion de predicciones")
    st.dataframe(
        {
            settings.prediction_column: list(summary.prediction_distribution.keys()),
            "count": list(summary.prediction_distribution.values()),
        },
        use_container_width=True,
    )

    st.subheader("Vista previa")
    st.dataframe(output_df.head(20), use_container_width=True)

    st.download_button(
        label="Descargar CSV con predicciones",
        data=dataframe_to_csv_bytes(output_df),
        file_name="predictions.csv",
        mime="text/csv",
    )


def _render_quiz_tab(model: Any, expected_columns: list[str]) -> None:
    st.subheader("Seccion 2: Quiz Pokemon")
    st.write("Completa tus stats y descubre tu Top-3 de Pokemon mas probables.")

    categorical_options = _extract_categorical_options(model)

    type_1_options = categorical_options.get(
        "Type 1",
        ["Bug", "Dragon", "Electric", "Fairy", "Fire", "Ghost", "Grass", "Psychic", "Water"],
    )
    type_2_options = ["(None)"] + categorical_options.get(
        "Type 2",
        ["Flying", "Poison", "Psychic", "Steel", "Water"],
    )
    legendary_options = categorical_options.get("Legendary", ["False", "True"])

    with st.form("pokemon_quiz_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            type_1 = st.selectbox("Type 1", type_1_options, index=0)
            type_2 = st.selectbox("Type 2", type_2_options, index=0)
            legendary = st.selectbox(
                "Legendary",
                legendary_options,
                index=legendary_options.index("False") if "False" in legendary_options else 0,
            )

        with col_b:
            total = st.slider("Total", **NUMERIC_FEATURE_CONFIG["Total"])
            hp = st.slider("HP", **NUMERIC_FEATURE_CONFIG["HP"])
            attack = st.slider("Attack", **NUMERIC_FEATURE_CONFIG["Attack"])
            defense = st.slider("Defense", **NUMERIC_FEATURE_CONFIG["Defense"])

        col_c, col_d = st.columns(2)
        with col_c:
            sp_atk = st.slider("Sp. Atk", **NUMERIC_FEATURE_CONFIG["Sp. Atk"])
            sp_def = st.slider("Sp. Def", **NUMERIC_FEATURE_CONFIG["Sp. Def"])
        with col_d:
            speed = st.slider("Speed", **NUMERIC_FEATURE_CONFIG["Speed"])
            generation = st.slider("Generation", **NUMERIC_FEATURE_CONFIG["Generation"])

        submit_quiz = st.form_submit_button("Descubrir mi Pokemon")

    if not submit_quiz:
        return

    quiz_input = {
        "Type 1": type_1,
        "Type 2": None if type_2 == "(None)" else type_2,
        "Total": total,
        "HP": hp,
        "Attack": attack,
        "Defense": defense,
        "Sp. Atk": sp_atk,
        "Sp. Def": sp_def,
        "Speed": speed,
        "Generation": generation,
        "Legendary": legendary,
    }

    try:
        quiz_df = pd.DataFrame([quiz_input])
        top_matches = generate_top_k_predictions(
            model=model,
            input_df=quiz_df,
            expected_columns=expected_columns,
            top_k=3,
        )
    except ValidationError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"No se pudo evaluar el quiz: {exc}")
        return

    if not top_matches:
        st.warning("No se encontraron coincidencias para el perfil ingresado.")
        return

    best_match = top_matches[0]
    st.success(f"Tu Pokemon mas probable es: {best_match.label} ({best_match.probability:.1%})")

    st.subheader("Top-3 coincidencias")
    ranking_df = pd.DataFrame(
        {
            "rank": [index for index in range(1, len(top_matches) + 1)],
            "pokemon": [item.label for item in top_matches],
            "probabilidad": [f"{item.probability:.1%}" for item in top_matches],
        }
    )
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    with st.expander("Ver perfil ingresado"):
        st.dataframe(quiz_df, use_container_width=True, hide_index=True)


def main() -> None:
    settings = get_settings()

    st.title("Pokemon ML Production Demo")
    st.write("Demo para produccion ML: prediccion batch por CSV + quiz interactivo.")

    try:
        model = cached_model(str(settings.model_path))
    except ModelLoadError as exc:
        st.error(str(exc))
        st.info("Entrena el modelo con el notebook y guarda el pipeline en models/pokemon_knn_pipeline.joblib.")
        return

    expected_columns = _resolve_expected_columns(settings, model)

    with st.sidebar:
        st.header("Configuracion")
        st.caption("Modelo")
        st.code(str(settings.model_path))
        st.caption("Columnas esperadas")
        if expected_columns:
            st.write(", ".join(expected_columns))
        else:
            st.warning("No se pudieron resolver columnas esperadas.")

    batch_tab, quiz_tab = st.tabs(["Prediccion batch", "Quiz Pokemon"])

    with batch_tab:
        _render_batch_tab(model=model, settings=settings, expected_columns=expected_columns)

    with quiz_tab:
        _render_quiz_tab(model=model, expected_columns=expected_columns)


if __name__ == "__main__":
    main()
