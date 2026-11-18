"""
API de predicción con FastAPI.

Uso, desde la carpeta src/:
    uvicorn model_deploy:app --host 0.0.0.0 --port 8000

Documentación interactiva: http://localhost:8000/docs
"""
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import ft_engineering
from model_evaluation import load_model

UMBRAL = 0.5

app = FastAPI(
    title="API Credit Scoring",
    description="Predice la probabilidad de que un nuevo cliente pague su crédito a tiempo.",
    version="1.0.0",
)

model = load_model()


class Cliente(BaseModel):
    """Datos de un crédito. Los campos opcionales que falten se imputan en el pipeline."""
    tipo_credito: int = Field(..., examples=[4])
    capital_prestado: float = Field(..., examples=[1500000])
    plazo_meses: int = Field(..., examples=[12])
    edad_cliente: int = Field(..., examples=[45])
    tipo_laboral: str = Field(..., examples=["Empleado"])
    salario_cliente: float = Field(..., examples=[3000000])
    total_otros_prestamos: float = Field(..., examples=[500000])
    cuota_pactada: float = Field(..., examples=[150000])
    puntaje_datacredito: Optional[float] = Field(None, examples=[790])
    cant_creditosvigentes: int = Field(..., examples=[3])
    huella_consulta: int = Field(..., examples=[4])
    saldo_mora: Optional[float] = Field(None, examples=[0])
    saldo_total: Optional[float] = Field(None, examples=[2000000])
    saldo_principal: Optional[float] = Field(None, examples=[1800000])
    saldo_mora_codeudor: Optional[float] = Field(None, examples=[0])
    creditos_sectorFinanciero: int = Field(..., examples=[2])
    creditos_sectorCooperativo: int = Field(..., examples=[0])
    creditos_sectorReal: int = Field(..., examples=[1])
    promedio_ingresos_datacredito: Optional[float] = Field(None, examples=[2800000])
    tendencia_ingresos: Optional[str] = Field(None, examples=["Creciente"])


class Lote(BaseModel):
    batch: List[Cliente]


def _predecir(clientes: List[Cliente]) -> List[float]:
    X = pd.DataFrame([c.model_dump() for c in clientes])
    X = ft_engineering.clean_features(X[ft_engineering.FEATURES])
    return model.predict_proba(X)[:, 1].round(4).tolist()


@app.get("/health")
def health():
    return {"status": "ok", "modelo": model.steps[-1][1].__class__.__name__}


@app.post("/predict")
def predict(cliente: Cliente):
    proba = _predecir([cliente])[0]
    return {
        "probabilidad_pago_a_tiempo": proba,
        "prediccion": "paga a tiempo" if proba >= UMBRAL else "riesgo de no pago",
    }


@app.post("/predict_batch")
def predict_batch(lote: Lote):
    if not lote.batch:
        raise HTTPException(status_code=422, detail="El lote está vacío.")
    # "predictions" es la probabilidad de pagar a tiempo de cada cliente
    return {"predictions": _predecir(lote.batch)}
