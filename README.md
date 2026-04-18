# ¡Deja de usar notebooks! Prepara tus modelos para funcionar en el mundo real

Proyecto demo para una charla universitaria sobre **ML en producción**.

La idea es mostrar, de punta a punta, cómo pasar de un notebook a un servicio desplegable:

1. Entrenar un modelo clásico (KNN) para predecir qué Pokémon eres.
2. Empaquetar todo en una app con Streamlit.
3. Contenerizar con Docker.
4. Publicar imagen en Google Artifact Registry.
5. Desplegar en Google Cloud Run.

## Estado actual

Implementado en esta versión:

- Pipeline de entrenamiento en notebook: `notebooks/01_train_model.ipynb`.
- Artefactos del modelo ya generados en `models/`.
- Frontend Streamlit para **predicción batch por CSV**.
- Validación de entrada, resumen de predicciones y descarga del CSV resultado.
- Dockerfiles (dev/prod), `cloudbuild.yaml` y `service.yaml` para despliegue.
- Tests unitarios base para validación/predicción.

Pendiente para la demo final:

- Segunda sección tipo **Quiz interactivo** (inputs manuales + top 3 candidatos).

## Estructura del proyecto

```text
ml-production-app/
├── data/
│   ├── raw/
│   └── sample/
├── models/
│   ├── pokemon_knn_pipeline.joblib
│   ├── model.pkl
│   └── pokemon_metadata.json
├── notebooks/
│   └── 01_train_model.ipynb
├── src/
│   ├── app.py
│   ├── config.py
│   ├── features.py
│   ├── model_io.py
│   ├── predict.py
│   └── validation.py
├── tests/
├── Dockerfile.dev
├── Dockerfile.prod
├── cloudbuild.yaml
├── service.yaml
├── commands.md
└── README.md
```

## Modelo y features

Notebook principal: `notebooks/01_train_model.ipynb`.

- Target: `Name`
- Modelo: `KNeighborsClassifier` (`n_neighbors=3`, `weights="distance"`)
- Artefactos exportados:
  - `models/pokemon_knn_pipeline.joblib`
  - `models/model.pkl`
- Metadata: `models/pokemon_metadata.json`

Features de entrada esperadas por el pipeline:

- `Type 1`
- `Type 2`
- `Total`
- `HP`
- `Attack`
- `Defense`
- `Sp. Atk`
- `Sp. Def`
- `Speed`
- `Generation`
- `Legendary`

## Requisitos

- Python 3.11
- `pip`
- Docker (opcional, para contenedores)
- Google Cloud SDK (opcional, para despliegue)

## Setup local

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
cp .env.example .env
```

Variables de entorno recomendadas para la demo Pokémon:

```bash
export MODEL_PATH=models/pokemon_knn_pipeline.joblib
export EXPECTED_COLUMNS="Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary"
export PREDICTION_COLUMN=prediction
```

## Ejecutar la app

```bash
streamlit run src/app.py
```

Funcionalidad actual en UI:

- Carga de `.csv`
- Validación de columnas
- Predicción batch
- Métricas rápidas:
  - filas procesadas
  - columnas recibidas
  - cantidad de clases predichas
- Distribución de predicciones
- Descarga de `predictions.csv`

## Formato de entrada CSV

El CSV debe contener las columnas esperadas por el modelo.

Ejemplo mínimo de cabecera:

```csv
Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary
```

Nota: los archivos en `data/sample/*.csv` son genéricos de scaffold y se pueden reemplazar por muestras Pokémon reales para la demo.

## Entrenamiento / reentrenamiento

Ejecuta:

```bash
jupyter lab notebooks/01_train_model.ipynb
```

El notebook entrena un `Pipeline` completo (preprocesamiento + modelo) y lo exporta. Eso evita duplicar lógica de transformación en producción.

## Tests

```bash
pytest
```

Cobertura actual orientada a:

- validación de CSV
- validación de columnas
- flujo de predicción y serialización de salida

## Docker

Build de imagen de producción:

```bash
docker build -f Dockerfile.prod -t ml-prediction-app:latest .
```

Run local de contenedor:

```bash
docker run --rm -p 4000:4000 \
  -e MODEL_PATH=models/pokemon_knn_pipeline.joblib \
  -e EXPECTED_COLUMNS="Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary" \
  -e PREDICTION_COLUMN=prediction \
  ml-prediction-app:latest
```

## Despliegue en Google Cloud

Este repo ya incluye lo necesario para pipeline de imagen + deploy:

- `cloudbuild.yaml` para build/push en Artifact Registry.
- `service.yaml` para Cloud Run.
- `commands.md` con comandos paso a paso.

Variables sugeridas:

```bash
export GCP_PROJECT_ID=your-gcp-project-id
export GCP_REGION=us-central1
export ARTIFACT_REPOSITORY=ml-model-serving
export IMAGE_NAME=ml-prediction-app
export IMAGE_TAG=latest
export CLOUD_RUN_SERVICE=ml-prediction-service
```

Build/push con Cloud Build:

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _GCP_REGION="${GCP_REGION}",_ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY}",_IMAGE_NAME="${IMAGE_NAME}",_IMAGE_TAG="${IMAGE_TAG}" \
  .
```

Deploy en Cloud Run:

```bash
gcloud run deploy "${CLOUD_RUN_SERVICE}" \
  --image "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --region "${GCP_REGION}" \
  --platform managed \
  --port 4000 \
  --set-env-vars MODEL_PATH=models/pokemon_knn_pipeline.joblib,EXPECTED_COLUMNS="Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary",PREDICTION_COLUMN=prediction
```

## Seguridad y buenas prácticas

- No versionar credenciales ni llaves de servicio.
- Cargar solo modelos confiables (`.pkl`, `.pickle`, `.joblib`) generados por tu pipeline.
- Mantener entrenamiento y serving desacoplados.
- Tratar validación de entrada como primera línea de defensa.

## Roadmap corto para la charla

1. Implementar sección **Quiz** en Streamlit (inputs manuales).
2. Usar `predict_proba` para mostrar **Top-1 + Top-2 + Top-3** Pokémon.
3. Añadir explicación simple del resultado (features más influyentes o distancia en KNN).
4. Reemplazar `data/sample` por ejemplos Pokémon reales para demo en vivo.
5. Agregar smoke test del contenedor y checklist pre-charla (build + deploy + URL final).
