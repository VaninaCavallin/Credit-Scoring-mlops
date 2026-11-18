import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


TARGET = "Pago_atiempo"

# Numéricas: solo imputación
COLUMNAS_NUMERIC = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito"
]

# Categóricas nominales: imputación + OneHotEncoder
COLUMNAS_CATEGORIC = [
    "tipo_laboral",
    "tendencia_ingresos"
]

# Categóricas ordinales: imputación + OrdinalEncoder (orden inferido en fit)
COLUMNAS_ORDINAL = [
    "tipo_credito"
]

EXCLUIR_DE_X = [
    "fecha_prestamo",
    # puntaje separa perfectamente a quienes pagan de quienes no (AUC = 1.0):
    # todos los que pagaron tienen puntaje >= 63.8 y todos los que no, <= 62.7.
    # Se calcula con información del resultado (data leakage), por eso se excluye.
    "puntaje"
]

CATEGORIAS_TENDENCIA = ["Creciente", "Decreciente", "Estable"]

FEATURES = COLUMNAS_NUMERIC + COLUMNAS_CATEGORIC + COLUMNAS_ORDINAL


def get_feature_columns(df: pd.DataFrame) -> tuple[list, list, list]:
    """
    Obtiene listas de columnas por tipo presentes en el df.
    """
    numeric = [c for c in COLUMNAS_NUMERIC if c in df.columns]
    cat_ = [c for c in COLUMNAS_CATEGORIC if c in df.columns]
    ord_ = [c for c in COLUMNAS_ORDINAL if c in df.columns]
    return numeric, cat_, ord_


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza mínima que deben recibir los datos antes del preprocesador,
    tanto en el entrenamiento como en la API:
    - tendencia_ingresos: los valores numéricos erróneos pasan a NaN.
    - categóricas: se convierten a texto, conservando los NaN para que los impute el pipeline.
    """
    X = X.copy()
    if "tendencia_ingresos" in X.columns:
        X["tendencia_ingresos"] = X["tendencia_ingresos"].where(
            X["tendencia_ingresos"].isin(CATEGORIAS_TENDENCIA)
        )
    for col in COLUMNAS_CATEGORIC + COLUMNAS_ORDINAL:
        if col in X.columns:
            X[col] = X[col].astype(object).where(X[col].isna(), X[col].astype(str))
    return X


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa X (solo las variables del modelo, ya limpias) e y.
    """
    X = clean_features(df[[c for c in FEATURES if c in df.columns]])
    y = df[TARGET]
    return X, y


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    División train/test estratificada (la clase "no paga a tiempo" es solo el 4.7%).
    Se usa la misma semilla en entrenamiento, evaluación y monitoreo.
    """
    X, y = prepare_data(df)
    return train_test_split(X, y, test_size=test_size, random_state=1234, stratify=y)


def build_preprocessor(numeric_cols, categorical_cols, ordinal_cols):
    """
    Construye el ColumnTransformer según el esquema:
    - numeric: SimpleImputer
    - categorical: SimpleImputer + OneHotEncoder
    - ordinal: SimpleImputer + OrdinalEncoder
    """

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_cols
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "categoric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first",
                        sparse_output=False
                    ))
                ]),
                categorical_cols
            )
        )

    if ordinal_cols:
        transformers.append(
            (
                "ordinal",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    ))
                ]),
                ordinal_cols
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )