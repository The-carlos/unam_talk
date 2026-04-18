# Deployment Commands

Estos comandos usan placeholders genericos. Reemplaza los valores antes de ejecutar.

```bash
export GCP_PROJECT_ID=your-gcp-project-id
export GCP_REGION=us-central1
export ARTIFACT_REPOSITORY=ml-model-serving
export IMAGE_NAME=ml-prediction-app
export IMAGE_TAG=latest
export CLOUD_RUN_SERVICE=ml-prediction-service
```

## 1. Configurar proyecto

```bash
gcloud config set project "${GCP_PROJECT_ID}"
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com
```

## 2. Crear repositorio en Artifact Registry

```bash
gcloud artifacts repositories create "${ARTIFACT_REPOSITORY}" \
  --repository-format=docker \
  --location="${GCP_REGION}" \
  --description="Docker images for ML model serving"
```

## 3. Construir imagen localmente

```bash
docker build -f Dockerfile.prod \
  -t "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}" \
  .
```

## 4. Autenticar Docker con Artifact Registry

```bash
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev"
```

## 5. Subir imagen

```bash
docker push "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
```

## 6. Construir y subir con Cloud Build

Antes de ejecutar este paso, asegurate de haber generado `models/pokemon_knn_pipeline.joblib` con el notebook. El modelo no debe versionarse en Git, pero si debe estar disponible en el contexto de build o descargarse desde un repositorio de artefactos adaptando el Dockerfile.

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _GCP_REGION="${GCP_REGION}",_ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY}",_IMAGE_NAME="${IMAGE_NAME}",_IMAGE_TAG="${IMAGE_TAG}" \
  .
```

## 7. Desplegar en Cloud Run con gcloud

```bash
gcloud run deploy "${CLOUD_RUN_SERVICE}" \
  --image "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --region "${GCP_REGION}" \
  --platform managed \
  --port 4000 \
  --set-env-vars MODEL_PATH=models/pokemon_knn_pipeline.joblib,EXPECTED_COLUMNS="Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary",PREDICTION_COLUMN=prediction
```

## 8. Desplegar con service.yaml

Actualiza la imagen en `service.yaml` con tus valores reales y ejecuta:

```bash
gcloud run services replace service.yaml --region "${GCP_REGION}"
```

## 9. Permitir acceso publico

Opcion con comando:

```bash
gcloud run services add-iam-policy-binding "${CLOUD_RUN_SERVICE}" \
  --region="${GCP_REGION}" \
  --member="allUsers" \
  --role="roles/run.invoker"
```

Opcion con archivo `gcr-service-policy.yaml`:

```bash
gcloud run services set-iam-policy "${CLOUD_RUN_SERVICE}" \
  gcr-service-policy.yaml \
  --region="${GCP_REGION}"
```
