# LATAM ML Challeenge

Autor: Santiago Aguirre — Senior Machine Learning Engineer

## 

## Setup inicial

- Migramos a `uv` para unificar dependencias en `pyproject.toml` y simplificar la gestión del entorno.
- Se mapearon `requirements.txt`, `requirements-dev.txt` y `requirements-test.txt` a dependencias base y extras (`dev`, `test`).
- Fijamos Python 3.11 (`.python-version` y `requires-python`) por compatibilidad con `numpy==1.22.4`.
- Agregamos `ruff` en dependencias de desarrollo para formateo y linting.
