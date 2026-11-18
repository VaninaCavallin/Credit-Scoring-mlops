"""
Entrena los modelos candidatos, elige el mejor por ROC-AUC en test y lo guarda
(junto con el preprocesador) en models/credit_model.joblib.

Uso, desde la carpeta src/:
    python train_model.py
"""
import json
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import carga_datos
import ft_engineering
from model_evaluation import compute_metrics

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "credit_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def build_candidates(y_train):
    # La clase 0 (no paga a tiempo) es minoritaria: se le da más peso en el entrenamiento
    ratio_0_1 = (y_train == 0).sum() / (y_train == 1).sum()
    return {
        "logistic": [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ],
        "random_forest": [
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5,
                class_weight="balanced", n_jobs=-1, random_state=1234,
            )),
        ],
        "xgboost": [
            ("model", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=ratio_0_1, eval_metric="auc", random_state=1234,
            )),
        ],
    }


def main():
    df = carga_datos.cargarDatos()
    X_train, X_test, y_train, y_test = ft_engineering.split_data(df)
    num, cat, ord_ = ft_engineering.get_feature_columns(X_train)

    results = {}
    best_name, best_pipe, best_auc = None, None, -1

    for name, steps in build_candidates(y_train).items():
        pipe = Pipeline([("preprocessor", ft_engineering.build_preprocessor(num, cat, ord_))] + steps)
        pipe.fit(X_train, y_train)
        metrics = compute_metrics(y_test, pipe.predict_proba(X_test)[:, 1])
        results[name] = metrics
        print(f"{name:<14} ROC-AUC={metrics['roc_auc']:.3f}  "
              f"Recall no pago={metrics['recall_no_pago']:.3f}  "
              f"Precision no pago={metrics['precision_no_pago']:.3f}")
        if metrics["roc_auc"] > best_auc:
            best_name, best_pipe, best_auc = name, pipe, metrics["roc_auc"]

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"\nMejor modelo: {best_name} (ROC-AUC={best_auc:.3f})")
    print(f"Guardado en {os.path.normpath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
