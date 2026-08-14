"""Configuración de la aplicación vía variables de entorno.

Se usa `pydantic-settings` en vez de constantes hardcodeadas porque el mismo
código corre en 3 contextos distintos con distinta configuración:
local (docker-compose), CI (tests) y Cloud Run (producción). Cloud Run
inyecta la variable de entorno `PORT` automáticamente y espera que el
contenedor escuche en ese puerto exacto -por eso `port` se lee de ahí y no
se hardcodea a 8000.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "churn-api"
    app_version: str = "1.0.0"
    model_path: str = "models/model.joblib"
    metadata_path: str = "models/metadata.json"
    # Cloud Run inyecta PORT en runtime; 8080 es el default de Cloud Run.
    port: int = 8080
    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
