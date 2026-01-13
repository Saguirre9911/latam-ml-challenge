# LATAM ML Challenge

Autor: Santiago Aguirre — Senior Machine Learning Engineer

## Setup inicial

- Migramos a `uv` para unificar dependencias en `pyproject.toml` y simplificar el setup local.
- Se mapearon `requirements.txt`, `requirements-dev.txt` y `requirements-test.txt` a dependencias base y extras (`dev`, `test`) para mantener compatibilidad con el `Makefile`.
- Fijamos Python 3.11 (`.python-version` y `requires-python`) por compatibilidad del stack cientifico; versiones antiguas fallaban al compilar/importar.
- Ajustamos `numpy` a `1.26.4` porque `mlflow` requiere `numpy<2`, manteniendo consistencia entre `pyproject.toml` y `requirements.txt`.
- Agregamos `ruff` en dependencias de desarrollo para formateo y linting.
- MLflow se incluye solo en `dev` para no afectar el runtime de la API.
- `xgboost` se movio a `dev` para reducir el tamano de la imagen de runtime.

## Parte 1 - Modelado

- Feature engineering segun el notebook: `period_day`, `high_season`, `min_diff` y `delay`.
- Features finales (top 10) definidas por importancia del modelo: `OPERA_Latin American Wings`, `MES_7`, `MES_10`, `OPERA_Grupo LATAM`, `MES_12`, `TIPOVUELO_I`, `MES_4`, `MES_11`, `OPERA_Sky Airline`, `OPERA_Copa Air`.
- Modelo elegido: Logistic Regression con balanceo de clases (mejor recall de clase 1 sin perder generalizacion).
- Balanceo implementado con `class_weight` (misma logica del notebook) para elevar recall y f1 de clase 1, que es el objetivo operacional.
- `DelayModel` incluye: preprocesamiento reproducible, entrenamiento y prediccion tipada; el modelo se guarda en `self._model`.
- MLflow opcional via `ENABLE_MLFLOW=1` para registrar parametros del entrenamiento sin afectar tests.
- Tests: `uv run pytest ... tests/model` (4/4 OK). Warning menor por tipos mixtos en `data.csv`.

## Parte 2 - API

- Implementada API con FastAPI en `challenge/api.py` con validacion de `OPERA`, `TIPOVUELO` y `MES` contra `data/data.csv`.
- Endpoint `POST /predict` usa `DelayModel` para preprocesar y predecir (sin necesidad de `Fecha-I`/`Fecha-O` en serving).
- Se agregaron modelos Pydantic para validar el payload y devolver `{"predict": [...]}`.
- Docker: `Dockerfile` y `docker-compose.yml` para levantar el servicio localmente.
- Validacion local: `/health` y `/predict` responden OK en localhost.
- Tests de API requieren `httpx` (agregado a `requirements-test.txt`).

## Parte 3 - Deploy

- Eleccion del servicio: **Cloud Run** por despliegue serverless, escalado a cero y costo bajo para demo (pago por uso).
- Registro de imagen: **Artifact Registry** para versionar la imagen Docker y mantener el flujo CI/CD estandar.
- Region `us-central1` para disponibilidad estable y latencia razonable en pruebas.
- Seguridad: `--allow-unauthenticated` solo para el demo; en prod se recomendaria auth y VPC.
- Ajuste critico: `exec format error` por build ARM local; se forzo `linux/amd64` con `docker buildx`.
- Optimización: `xgboost` movido a `dev` para reducir tamaño de imagen y tiempo de build.
- URL del servicio: `https://latam-challenge-api-127856830294.us-central1.run.app`.
- `STRESS_URL` actualizado en `Makefile`.

## Parte 3.1 - Stress Test

- Herramienta: Locust (headless), 60s, 100 usuarios, spawn rate 1.
- Endpoint probado: `POST /predict`.
- Resultado: 7,969 requests, 0% fallas.
- Latencias: p95 ~380ms, p99 ~550ms, max ~1.3s.

## Parte 4 - CI/CD

- Se implemento CI/CD en `.github/workflows` a partir de los archivos base.
- CI: instala dependencias con `uv`, corre `ruff`, `pytest tests/model` y `pytest tests/api`.
- CD: build/push de imagen a Artifact Registry y deploy a Cloud Run en `main`.
- Disparadores: `push`, `pull_request` y `workflow_dispatch` para ejecucion manual.
- Secrets requeridos (GitHub Actions): `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_AR_REPO`, `GCP_SERVICE_NAME`, `GCP_SA_KEY`.
- `GCP_SA_KEY` se genera desde un service account con roles: Cloud Run Admin, Artifact Registry Writer y Service Account User.
