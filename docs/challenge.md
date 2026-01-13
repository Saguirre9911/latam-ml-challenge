# LATAM ML Challenge

Autor: Santiago Aguirre — Senior Machine Learning Engineer

## Setup inicial

- Migramos a `uv` para unificar dependencias en `pyproject.toml` y simplificar la gestión del entorno.
- Se mapearon `requirements.txt`, `requirements-dev.txt` y `requirements-test.txt` a dependencias base y extras (`dev`, `test`).
- Fijamos Python 3.11 (`.python-version` y `requires-python`) por compatibilidad del stack científico.
- Actualizamos librerías porque versiones antiguas fallaban al compilar/importar en Python 3.11 (incompatibilidad binaria).
- Agregamos `ruff` en dependencias de desarrollo para formateo y linting.

## Parte 1 - Modelado

- Feature engineering: `period_day`, `high_season`, `min_diff` y `delay` (segun el notebook).
- Features finales (top 10): `OPERA_Latin American Wings`, `MES_7`, `MES_10`, `OPERA_Grupo LATAM`, `MES_12`, `TIPOVUELO_I`, `MES_4`, `MES_11`, `OPERA_Sky Airline`, `OPERA_Copa Air`.
- Modelo elegido: Logistic Regression con balanceo de clases (mejor recall de clase 1 sin perder generalizacion).

## Parte 2 - API

- Pendiente.

## Parte 3 - Deploy

- Pendiente.

## Parte 4 - CI/CD

- Pendiente.
