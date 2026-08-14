# churn-api — Servicio MLOps end-to-end de predicción de churn

Servicio de **predicción de fuga de clientes (customer churn)** para una entidad bancaria, expuesto como API REST. El modelo es una **regresión logística** entrenada sobre el dataset `Churn_Modelling.csv`, empaquetado con FastAPI, containerizado con Docker y desplegado en **Google Cloud Run** mediante un pipeline de CI/CD automatizado con GitHub Actions.

Proyecto desarrollado para la asignatura de MLOps (UAI) — pauta "Servicio MLOps end-to-end".

## Integrantes del equipo

- Alejandra Véliz
- Valentina Ariztía
- Leonor Saravia
- Jonathan Machuca
- Hemersson Gutiérrez

## 1. El problema: modelo de churn

Este servicio predice, para un cliente bancario dado, la **probabilidad de que abandone el banco (churn)**. La variable objetivo es `Exited` (1 = el cliente se fue, 0 = el cliente se quedó).

### Desbalance de clases

El dataset presenta un **desbalance de clases real**, calculado sobre las 10.000 filas de `Churn_Modelling.csv`:

| Clase | Significado | Cantidad | Porcentaje |
|---|---|---|---|
| 0 | Cliente se queda | 7.963 | **79.63%** |
| 1 | Cliente abandona (churn) | 2.037 | **20.37%** |

Este desbalance (~80/20) es la razón por la que el modelo se entrena con `class_weight="balanced"` y se evalúa con precision, recall, F1 y ROC-AUC — **no sólo con accuracy**, ya que un modelo trivial que siempre prediga "no churn" alcanzaría ~80% de accuracy sin ser útil para el negocio.

## 2. Fuente de datos y de código

La base de datos y el código utilizado como referencia para el modelo corresponden al obtenido de **[Customer Churn Dataset](https://www.kaggle.com/datasets/anandshaw2001/customer-churn-dataset)** (Kaggle), y al notebook público [ih8asham/customer-churn-dataset](https://www.kaggle.com/code/ih8asham/customer-churn-dataset), del cual se replicó fielmente el enfoque de preprocesamiento y feature engineering para la regresión logística (ver sección 4).

> **Privacidad:** este README y el modelo **no incluyen** `CustomerId`, `Surname`, ni ninguna otra clave o identificador personal del cliente. Estas columnas se descartan explícitamente en `training/data.py` (`ID_COLUMNS`) antes de que el modelo las vea.

## 3. Arquitectura y flujo end-to-end

```
Dataset (CSV) → training/ (entrenamiento + gate de calidad) → models/ (artifact versionado)
      → app/ (FastAPI, sirve el modelo) → Docker → Artifact Registry → Cloud Run
      → GitHub Actions orquesta todo el flujo en cada push a main
```

- **Entrenamiento** (`training/`): separado del servicio (`app/`). El modelo se entrena una vez, se serializa (`model.joblib`) y el servicio sólo lo carga y lo usa — nunca reentrena en caliente.
- **Servicio** (`app/`): FastAPI expone `/predict`, valida cada request contra un contrato Pydantic estricto, y nunca responde `500` genérico ante un input inválido (responde `422` con el detalle del campo).
- **Métricas de servicio** (bonus, ver sección 7): latencia, conteo de requests y errores expuestos en `/metrics` en formato Prometheus.
- **CI/CD** (`.github/workflows/ci-cd.yml`): 5 jobs encadenados como gates secuenciales — `lint → test → train-and-gate → docker-build → deploy`. Si cualquiera falla, los siguientes no se ejecutan.
- **Despliegue**: Google Cloud Run, contenedor sin estado (stateless), escala a cero cuando no hay tráfico.

## 4. Modelo: regresión logística

Pipeline fiel al notebook de referencia, implementado en `training/model.py`:

1. **Ingeniería de features**: se deriva `AgeGroup` a partir de `Age` con los mismos bins del notebook (`[18,30,50,100]` → `Young/Adult/Old`), calculado dentro del pipeline (no antes del split) para que se aplique igual en entrenamiento y en cada predicción individual servida por la API.
2. **Preprocesamiento**: `ColumnTransformer` con `OneHotEncoder(drop="first", handle_unknown="ignore")` para `Geography`, `Gender`, `AgeGroup`, y `StandardScaler` para las variables numéricas.
3. **Modelo base**: `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`.

**Features utilizadas:** `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` (+ `AgeGroup` derivado).
**Columnas excluidas siempre:** `RowNumber`, `CustomerId`, `Surname` (identificadores, sin valor predictivo y con datos personales).

### Desviaciones documentadas respecto al notebook original

Se declaran explícitamente por honestidad técnica, no son errores del notebook sino mejoras necesarias para servir el modelo como API:

| Notebook original | Este proyecto | Por qué |
|---|---|---|
| `pd.get_dummies` sobre todo el DataFrame | `ColumnTransformer` + `OneHotEncoder` fitteado en train | `get_dummies` genera columnas distintas según qué categorías aparecen en cada request; se rompe con una sola fila de inferencia (training/serving skew). |
| `train_test_split` sin `stratify` | `stratify=y` | El dataset tiene desbalance de clases; sin estratificar, el split puede generar folds de test con proporción de churn distinta a la real. |
| `LogisticRegression()` sin ponderar | `class_weight="balanced"` | Evita que el modelo colapse prediciendo siempre "no churn" dado el desbalance ~80/20. |
| `AgeGroup` calculado sobre todo el DataFrame antes del split | Calculado dentro del pipeline (`FunctionTransformer`) | Garantiza que la misma transformación se aplique en cada predicción de la API sobre una fila cruda, sin duplicar lógica. |

### Métricas reales obtenidas (test set, 2.000 filas, 20% del dataset)

Resultado de ejecutar `python -m training.train` sobre el dataset real:

| Métrica | Valor |
|---|---|
| Accuracy | 0.7115 |
| Precision (clase churn) | 0.3861 |
| Recall (clase churn) | 0.7076 |
| F1-score (clase churn) | 0.4996 |
| **ROC-AUC** | **0.7770** |

Matriz de confusión: TN=1135, FP=458, FN=119, TP=288.

El pipeline de CI/CD exige `ROC-AUC >= 0.75` para que el modelo pase el gate de calidad (`training/evaluate.py::summarize_for_gate`) — este modelo lo supera (0.777).

## 5. Estructura del proyecto

```
churn-mlops-service/
├── .github/workflows/ci-cd.yml   # Pipeline CI/CD (5 jobs)
├── app/
│   ├── main.py                   # Endpoints FastAPI
│   ├── schemas.py                # Contratos Pydantic (request/response)
│   ├── predictor.py              # Carga del modelo + lógica de predicción
│   ├── config.py                 # Configuración vía variables de entorno
│   └── metrics.py                # Métricas de servicio (Prometheus)
├── training/
│   ├── data.py                   # Carga, limpieza y split del dataset
│   ├── model.py                  # Definición del pipeline de modelado
│   ├── train.py                  # Script de entrenamiento + gate de calidad
│   └── evaluate.py                # Cálculo de métricas
├── models/
│   ├── model.joblib               # Modelo entrenado (artifact versionado)
│   └── metadata.json              # Métricas, balance de clases, versión
├── data/
│   └── Churn_Modelling.csv        # Dataset original
├── tests/                         # 25 tests (unit + integración)
├── docs/
│   └── informe.pdf                # Informe del proyecto (entregable)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt / requirements-dev.txt
├── pyproject.toml                 # Configuración de ruff y pytest
└── README.md
```

## 6. Cómo correr el proyecto

### Local (sin Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m training.train              # entrena y guarda models/model.joblib
uvicorn app.main:app --reload --port 8080
```

### Con Docker (recomendado, replica el entorno de producción)

```bash
docker compose up --build
```

Prueba real ejecutada durante el desarrollo (servidor local, mismo código que corre en el contenedor):

```
GET  /health   -> {"status":"ok","model_loaded":true,"model_version":"2026-08-14T01:24:50..."}
POST /predict  -> {"churn_prediction":0,"churn_probability":0.3564,"risk_level":"medium",...}
POST /predict (input inválido, Geography="Chile") -> HTTP 422 (nunca 500)
```

### Tests y calidad de código

```bash
ruff check .      # 0 errores
pytest             # 25 passed
```

## 7. Métricas de servicio expuestas (bonus)

El endpoint `GET /metrics` expone métricas en formato Prometheus, instrumentadas con `prometheus-fastapi-instrumentator`:

- `http_requests_total{method,handler,status}` — conteo de requests por endpoint y código de estado.
- `http_request_duration_seconds{method,handler}` — histograma de latencia por endpoint.
- `http_requests_inprogress{method,handler}` — requests en curso.
- `churn_predictions_total{prediction}` — contador de negocio: cuántas predicciones se hicieron y con qué resultado (0/1).
- `churn_prediction_errors_total` — errores al predecir.

Ejemplo real capturado en local:

```
churn_predictions_total{prediction="0"} 1.0
http_requests_total{handler="/predict",method="POST",status="2xx"} 1.0
http_requests_total{handler="/predict",method="POST",status="4xx"} 1.0
```

## 8. CI/CD (GitHub Actions)

`.github/workflows/ci-cd.yml` encadena 5 jobs como gates secuenciales:

1. **lint** — `ruff check .`
2. **test** — entrena un modelo de smoke y corre los 25 tests (`pytest`)
3. **train-and-gate** — reentrena con datos reales y **bloquea el pipeline** si `ROC-AUC < 0.75`; publica `model.joblib`/`metadata.json` como artifact
4. **docker-build** — descarga el artifact gateado, construye la imagen y la publica en Artifact Registry (sólo en push a `main`, y sólo si el proyecto de GCP ya está configurado — variable de repo `GCP_PROJECT_ID`)
5. **deploy** — despliega a Cloud Run y corre un smoke test real contra `/health` en la URL pública

## 9. Despliegue en Google Cloud Run

Requiere: proyecto de GCP con billing habilitado, Artifact Registry, una service account con roles `run.admin`, `artifactregistry.writer`, `iam.serviceAccountUser`, y sus credenciales guardadas en el secret `GCP_SA_KEY` del repositorio, además de las variables de repo `GCP_PROJECT_ID` y `GCP_REGION`.

```bash
gcloud run deploy churn-api \
  --image REGION-docker.pkg.dev/PROJECT_ID/churn-api/churn-api:latest \
  --region REGION \
  --allow-unauthenticated \
  --memory=512Mi --min-instances=0 --max-instances=3
```

Cloud Run inyecta la variable de entorno `PORT`; el `Dockerfile` respeta esa variable en tiempo de ejecución (`CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}`).
