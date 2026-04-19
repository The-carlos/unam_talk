"""Streamlit frontend for batch CSV predictions and interactive quiz."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd
import requests
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
}

FEATURE_LABELS_ES = {
    "Type 1": "Tipo principal",
    "Type 2": "Tipo secundario",
    "Total": "Puntaje total",
    "HP": "Puntos de salud (HP)",
    "Attack": "Ataque",
    "Defense": "Defensa",
    "Sp. Atk": "Ataque especial",
    "Sp. Def": "Defensa especial",
    "Speed": "Velocidad",
    "Legendary": "Es legendario",
}

FEATURE_HELP_ES = {
    "Type 1": "Cada Pokemon tiene un tipo principal que define fortalezas y debilidades.",
    "Type 2": "Algunos Pokemon tienen un segundo tipo.",
    "Total": "Suma total de stats; sirve como referencia general de fuerza.",
    "HP": "Puntos de salud: cuanto dano puede resistir antes de caer.",
    "Attack": "Modificador base para ataques normales.",
    "Defense": "Resistencia base contra ataques normales.",
    "Sp. Atk": "Modificador base para ataques especiales.",
    "Sp. Def": "Resistencia base contra ataques especiales.",
    "Speed": "Define que Pokemon ataca primero en cada turno.",
    "Legendary": "Indica si el Pokemon es legendario.",
}

POKEAPI_BASE_URL = "https://pokeapi.co/api/v2/pokemon"


@st.cache_resource(show_spinner=False)
def cached_model(model_path: str):
    return load_model(model_path)


def fetch_random_gen1_pokemon_sprite() -> dict[str, str] | None:
    pokemon_id = random.randint(1, 151)
    try:
        response = requests.get(f"{POKEAPI_BASE_URL}/{pokemon_id}", timeout=8)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return None

    name = str(payload.get("name", "")).strip()
    official_artwork = payload.get("sprites", {}).get("other", {}).get("official-artwork", {}).get("front_default")
    sprite = payload.get("sprites", {}).get("front_default")
    image_url = official_artwork or sprite
    if not name or not image_url:
        return None

    return {"name": name.title(), "image_url": str(image_url)}


def _pokemon_api_candidates(name: str) -> list[str]:
    cleaned = name.strip().lower()
    aliases = {
        "mr. mime": "mr-mime",
        "farfetch'd": "farfetchd",
        "nidoran♀": "nidoran-f",
        "nidoran♂": "nidoran-m",
    }

    candidates = [
        aliases.get(cleaned, cleaned),
        cleaned.replace(" ", "-"),
        cleaned.replace(" ", "-").replace(".", ""),
        cleaned.replace(" ", "-").replace(".", "").replace("'", ""),
    ]

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_pokemon_sprite_url(name: str) -> str | None:
    for candidate in _pokemon_api_candidates(name):
        url = f"{POKEAPI_BASE_URL}/{candidate}"
        try:
            response = requests.get(url, timeout=8)
        except requests.RequestException:
            continue

        if response.status_code == 404:
            continue
        if response.status_code != 200:
            continue

        payload = response.json()
        official_artwork = payload.get("sprites", {}).get("other", {}).get("official-artwork", {}).get("front_default")
        if official_artwork:
            return official_artwork

        sprite = payload.get("sprites", {}).get("front_default")
        if sprite:
            return sprite

    return None


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
            type_1 = st.selectbox(FEATURE_LABELS_ES["Type 1"], type_1_options, index=0, help=FEATURE_HELP_ES["Type 1"])
            type_2 = st.selectbox(FEATURE_LABELS_ES["Type 2"], type_2_options, index=0, help=FEATURE_HELP_ES["Type 2"])
            legendary = st.selectbox(
                FEATURE_LABELS_ES["Legendary"],
                legendary_options,
                index=legendary_options.index("False") if "False" in legendary_options else 0,
                help=FEATURE_HELP_ES["Legendary"],
            )

        with col_b:
            total = st.slider(FEATURE_LABELS_ES["Total"], help=FEATURE_HELP_ES["Total"], **NUMERIC_FEATURE_CONFIG["Total"])
            hp = st.slider(FEATURE_LABELS_ES["HP"], help=FEATURE_HELP_ES["HP"], **NUMERIC_FEATURE_CONFIG["HP"])
            attack = st.slider(
                FEATURE_LABELS_ES["Attack"], help=FEATURE_HELP_ES["Attack"], **NUMERIC_FEATURE_CONFIG["Attack"]
            )
            defense = st.slider(
                FEATURE_LABELS_ES["Defense"], help=FEATURE_HELP_ES["Defense"], **NUMERIC_FEATURE_CONFIG["Defense"]
            )

        col_c, col_d = st.columns(2)
        with col_c:
            sp_atk = st.slider(
                FEATURE_LABELS_ES["Sp. Atk"], help=FEATURE_HELP_ES["Sp. Atk"], **NUMERIC_FEATURE_CONFIG["Sp. Atk"]
            )
            sp_def = st.slider(
                FEATURE_LABELS_ES["Sp. Def"], help=FEATURE_HELP_ES["Sp. Def"], **NUMERIC_FEATURE_CONFIG["Sp. Def"]
            )
        with col_d:
            speed = st.slider(FEATURE_LABELS_ES["Speed"], help=FEATURE_HELP_ES["Speed"], **NUMERIC_FEATURE_CONFIG["Speed"])

        submit_quiz = st.form_submit_button("Descubrir mi Pokemon")

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
        "Generation": 1,
        "Legendary": legendary,
    }

    if not submit_quiz:
        return

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
    st.markdown(
        (
            "<div style='background:#dcfce7;border:1px solid #86efac;border-radius:10px;padding:20px;"
            "text-align:center;margin-bottom:14px;'>"
            "<div style='font-size:16px;font-weight:600;color:#14532d;'>Tu Pokemon mas probable es</div>"
            f"<div style='font-size:44px;line-height:1.1;font-weight:800;color:#14532d;'>{best_match.label}</div>"
            f"<div style='font-size:24px;font-weight:700;color:#166534;'>{best_match.probability:.1%}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    sprite_url = fetch_pokemon_sprite_url(best_match.label)
    if sprite_url:
        st.markdown(
            (
                "<div style='text-align:center;margin:8px 0 12px 0;'>"
                f"<img src='{sprite_url}' alt='{best_match.label}' style='width:260px;max-width:100%;'/>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.caption("No se encontro imagen para este Pokemon en PokeAPI.")

    st.subheader("Top-3 coincidencias")
    ranking_df = pd.DataFrame(
        {
            "rank": [index for index in range(1, len(top_matches) + 1)],
            "Pokemon": [item.label for item in top_matches],
            "Probabilidad": [f"{item.probability:.1%}" for item in top_matches],
        }
    )
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)


def main() -> None:
    settings = get_settings()

    if "header_pokemon" not in st.session_state:
        st.session_state["header_pokemon"] = fetch_random_gen1_pokemon_sprite()

    title_col, pokemon_col = st.columns([5, 1])
    with title_col:
        st.title("¿Quiénesese pokemoooon? (Pero con IA)")
        st.write("Usa los poderes de la IA para predecir quién es ese pokemos basado en sus características.")
    with pokemon_col:
        header_pokemon = st.session_state.get("header_pokemon")
        if header_pokemon:
            st.image(
                header_pokemon["image_url"],
                caption=header_pokemon["name"],
                use_container_width=True,
            )

    try:
        model = cached_model(str(settings.model_path))
    except ModelLoadError as exc:
        st.error(str(exc))
        st.info("Entrena el modelo con el notebook y guarda el pipeline en models/pokemon_knn_pipeline.joblib.")
        return

    expected_columns = _resolve_expected_columns(settings, model)

    with st.sidebar:
        st.header("Sobre Este Proyecto")
        st.write(
            "Demo de ML en produccion: entrenamos un modelo clasico para predecir tu Pokemon "
            "y lo llevamos a una app lista para desplegar."
        )
        st.markdown(
            "¡Te reto a desplegar este modelo por tu propia cuenta! "
            "Todo el código de este proyecto esta disponible en este repo:\n\n"
            "https://github.com/The-carlos/unam_talk"
        )
        st.markdown(
            "### Conectemos\n"
            "Gracias por pasar por la demo. Si te gustó, me dara mucho gusto que conectemos en redes:\n\n"
            "[¡Sigueme por acá!](https://linktr.ee/theCarlos?utm_source=linktree_profile_share&ltsid=b1b756c5-89af-4ad3-be61-5fc77dd11b87)"
        )
        st.divider()
        st.caption("Configuracion tecnica")
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
