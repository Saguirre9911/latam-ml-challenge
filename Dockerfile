# syntax=docker/dockerfile:1.2
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY challenge /app/challenge
COPY data /app/data

EXPOSE 8000

CMD ["/bin/sh", "-c", "uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
