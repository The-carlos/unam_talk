# ¡Deja de usar notebooks! Prepara tus modelos para funcionar en el mundo real

Demo de ML en produccion para charla universitaria.

El proyecto muestra un flujo end-to-end:

1. Entrenamiento en notebook.
2. Empaquetado del modelo serializado.
3. App de inferencia en Streamlit.
4. Contenerizacion con Docker.
5. Build y push con Cloud Build.
6. Despliegue declarativo en Cloud Run con `service.yaml`.

## Estado actual (actualizado: 2026-04-19)

Implementado y validado:

- Notebook de entrenamiento: `notebooks/01_train_model.ipynb`.
- Pipeline entrenado exportado en `models/pokemon_knn_pipeline.joblib`.
- App Streamlit funcional con 2 flujos:
  - Prediccion batch por CSV.
  - Quiz interactivo con Top-3 (`predict_proba`).
- Integracion con PokeAPI para mostrar sprite del Top-1 en quiz.
- Contenedores dev/prod listos (`Dockerfile.dev`, `Dockerfile.prod`).
- Build remoto en GCP exitoso con `cloudbuild.yaml`.
- Servicio desplegado en Cloud Run con `service.yaml`.
- Tests unitarios pasando (`8 passed`, pytest).

URL del ultimo despliegue validado:

- `https://ml-prediction-service-wxuhlxmv7a-uc.a.run.app`

## Estructura del repositorio

```text
.
├── .devcontainer/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── sample/
├── models/
├── notebooks/
├── src/
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

Features esperadas por inferencia:

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

## Aplicacion Streamlit

Archivo principal: `src/app.py`

### 1) Prediccion batch por CSV

- Carga de archivo `.csv`.
- Validacion de columnas.
- Prediccion del lote completo.
- Resumen de metricas y distribucion.
- Descarga de `predictions.csv`.

CSV de ejemplo:

- `data/sample/pokemon_batch_input_demo.csv`

### 2) Quiz Pokemon

- Formulario con features manuales.
- `Generation` fija en `1`.
- Top-3 resultados con probabilidad.
- Imagen del Top-1 consultada en PokeAPI.

## Configuracion por variables de entorno

Plantilla: `.env.example`

```bash
MODEL_PATH=models/pokemon_knn_pipeline.joblib
EXPECTED_COLUMNS=Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary
PREDICTION_COLUMN=prediction
```

Nota: el proyecto usa `os.getenv(...)`; `.env` no se carga automaticamente.

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

## Setup con Dev Container (recomendado para despliegue)

`Dockerfile.dev` instala `google-cloud-cli`, por lo que es el entorno ideal para `gcloud`.

Si `gcloud` no aparece:

1. VS Code: `Dev Containers: Rebuild and Reopen in Container`
2. Abre nueva terminal
3. Verifica:

```bash
gcloud --version
```

## Tests

```bash
pytest
```

Estado esperado actual: verde.

## Docker de produccion

Build local:

```bash
docker build -f Dockerfile.prod -t ml-prediction-app:latest .
```

Run local:

```bash
docker run --rm -p 4000:4000 \
  -e MODEL_PATH=models/pokemon_knn_pipeline.joblib \
  -e EXPECTED_COLUMNS="Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary" \
  -e PREDICTION_COLUMN=prediction \
  ml-prediction-app:latest
```

## Despliegue en GCP (flujo usado y validado)

### Contexto real usado

- Proyecto: `project-ai-engineering`
- Region: `us-central1`
- Artifact Registry repo: `repo-compu-fest-pokemon`
- Imagen: `image-v1-compu-fest-pokemon:latest`
- Servicio Cloud Run: `ml-prediction-service`

### 1) Crear repositorio Docker en Artifact Registry (si aun no existe)

```bash
gcloud artifacts repositories create repo-compu-fest-pokemon \
  --repository-format docker \
  --project project-ai-engineering \
  --location us-central1
```

### 2) Build y push de imagen con Cloud Build

`cloudbuild.yaml` ya apunta a la imagen real:

- `us-central1-docker.pkg.dev/project-ai-engineering/repo-compu-fest-pokemon/image-v1-compu-fest-pokemon:latest`

Ejecutar:

```bash
gcloud builds submit --config=cloudbuild.yaml --project project-ai-engineering
```

### 3) Despliegue declarativo con `service.yaml`

`service.yaml` documenta y define:

- nombre del servicio,
- politica de ingress,
- escalado min/max,
- concurrencia,
- imagen desplegada,
- puerto del contenedor,
- variables de entorno del modelo,
- limites de CPU/memoria.

Aplicar manifiesto:

```bash
gcloud run services replace service.yaml \
  --project project-ai-engineering \
  --region us-central1
```

### 4) Hacer publico el servicio

```bash
gcloud run services add-iam-policy-binding ml-prediction-service \
  --project project-ai-engineering \
  --region us-central1 \
  --member="allUsers" \
  --role="roles/run.invoker"
```

### 5) Obtener URL del servicio

```bash
gcloud run services describe ml-prediction-service \
  --project project-ai-engineering \
  --region us-central1 \
  --format='value(status.url)'
```

## Troubleshooting rapido

- Si Cloud Run no levanta la app, confirma que el contenedor escucha en el puerto esperado (`4000` en este proyecto).
- Si falla el modelo, valida que exista `models/pokemon_knn_pipeline.joblib` dentro de la imagen.
- Si PokeAPI falla, la app sigue funcionando pero puede no mostrar imagen del Pokemon.
- Cargar solo modelos `pickle/joblib` de fuentes confiables.

## Checklist de verificacion final

1. `pytest` en verde.
2. `gcloud builds submit ...` en `SUCCESS`.
3. `gcloud run services replace ...` sin errores.
4. URL de Cloud Run responde y carga Streamlit.
5. Flujo batch + quiz probado manualmente.

## Referencias

- Guia de comandos extendida: `commands.md`
- Politica IAM opcional: `gcr-service-policy.yaml`
- Repo: https://github.com/The-carlos/unam_talk
