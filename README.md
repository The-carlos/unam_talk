# ¡Deja de usar notebooks! Prepara tus modelos para funcionar en el mundo real

Proyecto demo para una charla universitaria sobre ML en produccion.

Objetivo: mostrar un flujo completo desde notebook hasta despliegue real.

1. Entrenar un modelo clasico (KNN) para predecir que Pokemon eres.
2. Servir el modelo en una app con Streamlit.
3. Contenerizar con Docker.
4. Publicar imagen en Google Artifact Registry.
5. Desplegar en Google Cloud Run.

## Estado actual (Abril 2026)

Implementado:

- Notebook de entrenamiento completo: `notebooks/01_train_model.ipynb`.
- Pipeline entrenado y exportado en `models/`.
- App Streamlit con 2 secciones:
  - Prediccion batch por CSV.
  - Quiz interactivo con Top-3 via `predict_proba`.
- Integracion con PokeAPI para mostrar imagen del Pokemon Top-1 en el quiz.
- Header con Pokemon aleatorio de Gen 1 al abrir/refrescar.
- Sidebar con descripcion del proyecto, reto de despliegue y links.
- Dockerfiles dev/prod, `cloudbuild.yaml`, `service.yaml`, `commands.md`.
- Tests unitarios de validacion/prediccion pasando.

## Estructura

```text
ml-production-app/
├── .devcontainer/
│   └── devcontainer.json
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
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Modelo

Notebook principal: `notebooks/01_train_model.ipynb`

- Target: `Name`
- Modelo: `KNeighborsClassifier(n_neighbors=3, weights="distance")`
- Artefactos:
  - `models/pokemon_knn_pipeline.joblib` (principal para inferencia)
  - `models/model.pkl` (compatibilidad)
  - `models/pokemon_metadata.json`

Features esperadas:

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

## Notebook: seccion extra KNN

Se agrego una seccion para visualizar vecinos mas cercanos del KNN:

- `## 7. Visualize nearest neighbors (KNN intuition)`

Esta seccion muestra:

- fila query de prueba,
- vecinos mas cercanos del train,
- distancia por vecino,
- barplot horizontal de cercania.

## App Streamlit

Archivo principal: `src/app.py`

### Seccion 1: Batch CSV

- Subida de CSV.
- Validacion de columnas y lectura segura.
- Prediccion batch.
- Resumen de metricas.
- Descarga de `predictions.csv`.

Archivo de muestra:

- `data/sample/pokemon_batch_input_demo.csv`

### Seccion 2: Quiz

- Inputs manuales para features.
- `Generation` fija en `1` (no seleccionable en UI).
- Resultado principal destacado (caja verde, texto centrado y grande).
- Top-3 coincidencias con probabilidad.
- Imagen del Pokemon Top-1 via PokeAPI.

## Variables de entorno

`.env.example` ya esta alineado al modelo Pokemon.

Ejemplo recomendado:

```bash
MODEL_PATH=models/pokemon_knn_pipeline.joblib
EXPECTED_COLUMNS="Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary"
PREDICTION_COLUMN=prediction
```

Importante: tu codigo usa `os.getenv(...)`, asi que `.env` no se carga solo.

Carga manual:

```bash
set -a
source .env
set +a
```

## Setup local (sin contenedor)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
cp .env.example .env
set -a && source .env && set +a
streamlit run src/app.py
```

## Setup con Dev Container (recomendado para deploy)

Raiz de repo: `ml-production-app/`

Config activa:

- `.devcontainer/devcontainer.json` (dentro de `ml-production-app`)
- build con `Dockerfile.dev`

`Dockerfile.dev` instala `google-cloud-cli`, por eso este entorno es el indicado para `gcloud`.

### Si `gcloud` no existe

Normalmente significa que no estas dentro del contenedor reconstruido.

Pasos:

1. VS Code: `Dev Containers: Rebuild and Reopen in Container`
2. Nueva terminal
3. Verificar:

```bash
gcloud --version
```

## Tests

```bash
pytest
```

Estado actual esperado: tests verdes.

## Docker (produccion)

Build:

```bash
docker build -f Dockerfile.prod -t ml-prediction-app:latest .
```

Run:

```bash
docker run --rm -p 4000:4000 \
  -e MODEL_PATH=models/pokemon_knn_pipeline.joblib \
  -e EXPECTED_COLUMNS="Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary" \
  -e PREDICTION_COLUMN=prediction \
  ml-prediction-app:latest
```

## Despliegue GCP

Referencia completa: `commands.md`

Variables sugeridas:

```bash
export GCP_PROJECT_ID=your-gcp-project-id
export GCP_REGION=us-central1
export ARTIFACT_REPOSITORY=ml-model-serving
export IMAGE_NAME=ml-prediction-app
export IMAGE_TAG=latest
export CLOUD_RUN_SERVICE=ml-prediction-service
```

Build + push con Cloud Build:

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _GCP_REGION="${GCP_REGION}",_ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY}",_IMAGE_NAME="${IMAGE_NAME}",_IMAGE_TAG="${IMAGE_TAG}" \
  .
```

Deploy Cloud Run:

```bash
gcloud run deploy "${CLOUD_RUN_SERVICE}" \
  --image "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --region "${GCP_REGION}" \
  --platform managed \
  --port 4000 \
  --set-env-vars MODEL_PATH=models/pokemon_knn_pipeline.joblib,EXPECTED_COLUMNS="Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary",PREDICTION_COLUMN=prediction
```

## Checklist de continuidad (post-rebuild)

Ejecuta esto despues del rebuild para continuar rapido:

1. Entrar al repo:

```bash
cd /workspaces/ml_for_production_talk/ml-production-app
```

2. Verificar entorno:

```bash
gcloud --version
python --version
```

3. Cargar variables:

```bash
set -a && source .env && set +a
```

4. Probar app:

```bash
streamlit run src/app.py
```

5. Probar tests:

```bash
pytest
```

6. Inicializar GCP:

```bash
gcloud init
gcloud auth application-default login
```

7. Seguir `commands.md` para Artifact Registry + Cloud Run.

## Riesgos / notas

- PokeAPI es dependencia externa de red (si falla, la app sigue pero puede no mostrar imagen).
- El modelo usa pickle/joblib: cargar solo artefactos confiables.
- No versionar secretos/credenciales.

## Repo

- GitHub: https://github.com/The-carlos/unam_talk
