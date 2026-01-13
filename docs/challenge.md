# LATAM ML Challenge

Autor: Santiago Aguirre — Senior Machine Learning Engineer

## Setup inicial

- Migramos a `uv` para unificar dependencias en `pyproject.toml` y simplificar el setup local.
- Se mapearon `requirements.txt`, `requirements-dev.txt` y `requirements-test.txt` a dependencias base y extras (`dev`, `test`) para mantener compatibilidad con el `Makefile`.
- Fijamos Python 3.11 (`.python-version` y `requires-python`) por compatibilidad del stack cientifico; versiones antiguas fallaban al compilar/importar.
- Ajustamos `numpy` a `1.26.4` porque `mlflow` requiere `numpy<2`, manteniendo consistencia entre `pyproject.toml` y `requirements.txt`.
- Agregamos `ruff` en dependencias de desarrollo para formateo y linting.
- MLflow se incluye solo en `dev` para no afectar el runtime de la API.

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

- Pendiente.

## Parte 4 - CI/CD

- Pendiente.
