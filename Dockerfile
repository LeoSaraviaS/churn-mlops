# --- Imagen base ------------------------------------------------------------
# python:3.12-slim: imagen oficial reducida (menor superficie de ataque y
# menor tamaño que la imagen completa), suficiente para correr scikit-learn.
FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema mínimas para compilar wheels de numpy/scikit-learn
# si no hay wheel binaria disponible para la arquitectura del build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar sólo requirements primero para aprovechar la cache de capas de Docker:
# si el código cambia pero no las dependencias, esta capa no se reconstruye.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación y el modelo ya entrenado.
# En CI, el job docker-build descarga el model.joblib recién entrenado y
# gateado por train-and-gate ANTES de este build, así la imagen siempre
# lleva el modelo más reciente que pasó el gate de calidad.
COPY app/ ./app/
COPY training/ ./training/
COPY models/ ./models/

# Usuario no-root: buena práctica de seguridad, evita que un proceso
# comprometido dentro del contenedor tenga privilegios de root.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Cloud Run inyecta la variable de entorno PORT (por defecto 8080) y espera
# que el contenedor escuche exactamente en ese puerto.
ENV PORT=8080
EXPOSE 8080

# HEALTHCHECK: permite a Docker (y a docker-compose) saber si el contenedor
# está realmente sirviendo tráfico, no sólo si el proceso sigue vivo.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f "http://localhost:${PORT}/health" || exit 1

# Forma shell (no exec) para que ${PORT} se expanda en runtime: Cloud Run
# asigna un puerto dinámico vía esta variable, así que no puede hardcodearse
# en forma exec (["uvicorn", ..., "--port", "8080"]).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
