# Imagen de la API de predicción (src/model_deploy.py)
# Requiere haber entrenado el modelo antes: python src/train_model.py
FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY models/credit_model.joblib models/credit_model.joblib

WORKDIR /app/src
EXPOSE 8000

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
