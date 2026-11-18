"""
Evaluación del modelo guardado en models/credit_model.joblib sobre el conjunto de test.

Uso, desde la carpeta src/:
    python model_evaluation.py
"""
import io
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "credit_model.joblib")


def compute_metrics(y_true, proba_pago, threshold: float = 0.5) -> dict:
    """
    Métricas de clasificación. proba_pago es la probabilidad de pagar a tiempo (clase 1).
    Precision, recall y F1 se calculan para la clase de interés: "no paga a tiempo" (clase 0).
    """
    y_true = np.asarray(y_true)
    proba_pago = np.asarray(proba_pago)
    y_pred = (proba_pago >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_no_pago": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_no_pago": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_no_pago": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba_pago),
        # PR-AUC de la clase minoritaria: más informativa que el accuracy con 4.7% de no pago
        "pr_auc_no_pago": average_precision_score(1 - y_true, 1 - proba_pago),
        "casos_no_pago_test": int((y_true == 0).sum()),
    }


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el modelo en {os.path.normpath(path)}. "
            "Ejecutar primero: python train_model.py"
        )
    return joblib.load(path)


def evaluation():
    """
    Evalúa el modelo guardado y devuelve las métricas y un PNG (BytesIO) con
    la matriz de confusión, la curva ROC y las métricas principales.
    """
    import carga_datos
    import ft_engineering

    model = load_model()
    df = carga_datos.cargarDatos()
    _, X_test, _, y_test = ft_engineering.split_data(df)

    proba_pago = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, proba_pago)
    y_pred = (proba_pago >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Evaluación del modelo {model.steps[-1][1].__class__.__name__} (test)", fontsize=18)

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No paga a tiempo", "Paga a tiempo"],
        cmap="cividis", ax=axes[0], colorbar=False,
    )
    axes[0].set_title("Matriz de confusión")

    RocCurveDisplay.from_predictions(y_test, proba_pago, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_title(f"Curva ROC (AUC = {metrics['roc_auc']:.3f})")

    custom_titles = {
        "precision_no_pago": "Precisión\nno pago",
        "recall_no_pago": "Recall\nno pago",
        "f1_no_pago": "F1\nno pago",
        "roc_auc": "ROC AUC",
        "pr_auc_no_pago": "PR AUC\nno pago",
    }
    axes[2].bar(custom_titles.values(), [metrics[k] for k in custom_titles], color="#2c5c8a")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Métricas principales")
    for i, k in enumerate(custom_titles):
        axes[2].text(i, metrics[k] + 0.02, f"{metrics[k]:.2f}", ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return metrics, buf


if __name__ == "__main__":
    matplotlib.use("Agg")
    metrics, buf = evaluation()
    for name, value in metrics.items():
        print(f"{name:<20}: {value:.3f}" if isinstance(value, float) else f"{name:<20}: {value}")

    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out = os.path.join(reports_dir, "evaluacion_modelo.png")
    with open(out, "wb") as f:
        f.write(buf.getvalue())
    print(f"\nGráfico guardado en {os.path.normpath(out)}")
