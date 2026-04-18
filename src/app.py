"""Streamlit frontend for batch CSV predictions."""

from __future__ import annotations

import streamlit as st

try:
    from src.config import get_settings
    from src.features import normalize_column_names
    from src.model_io import ModelLoadError, load_model
    from src.predict import dataframe_to_csv_bytes, generate_predictions
    from src.validation import ValidationError, read_csv_file
except ModuleNotFoundError:
    # Fallback when Streamlit executes from the src/ directory context.
    from config import get_settings
    from features import normalize_column_names
    from model_io import ModelLoadError, load_model
    from predict import dataframe_to_csv_bytes, generate_predictions
    from validation import ValidationError, read_csv_file


st.set_page_config(page_title="ML Batch Prediction", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_model(model_path: str):
    return load_model(model_path)


def main() -> None:
    settings = get_settings()

    st.title("Prediccion batch con modelo ML")
    st.write("Carga un CSV, valida sus columnas y descarga un archivo con predicciones.")

    with st.sidebar:
        st.header("Configuracion")
        st.caption("Modelo")
        st.code(str(settings.model_path))
        st.caption("Columnas esperadas")
        if settings.expected_columns:
            st.write(", ".join(settings.expected_columns))
        else:
            st.warning("EXPECTED_COLUMNS no esta configurado.")

    try:
        model = cached_model(str(settings.model_path))
    except ModelLoadError as exc:
        st.error(str(exc))
        st.info("Entrena el modelo con el notebook y guarda el pipeline en models/model.pkl.")
        return

    uploaded_file = st.file_uploader("Archivo CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Sube un archivo .csv para generar predicciones.")
        return

    try:
        input_df = normalize_column_names(read_csv_file(uploaded_file))
        output_df, summary = generate_predictions(
            model=model,
            input_df=input_df,
            expected_columns=settings.expected_columns,
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


if __name__ == "__main__":
    main()
