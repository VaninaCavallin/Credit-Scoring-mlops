# 📊 Credit Scoring MLOps Pipeline

> Pipeline de Machine Learning y MLOps para la predicción del comportamiento de nuevos clientes en una empresa financiera, abarcando desde la carga de datos hasta el despliegue y monitoreo del modelo.

---

# 📌 Descripción del proyecto

Las instituciones financieras necesitan evaluar el riesgo asociado a nuevos clientes para optimizar la concesión de créditos y reducir pérdidas.

Este proyecto desarrolla un pipeline completo de Machine Learning siguiendo principios de **MLOps**, incorporando buenas prácticas para la construcción, evaluación, despliegue y monitoreo de modelos predictivos.

Más allá del entrenamiento del modelo, el objetivo es demostrar cómo organizar un proyecto de Ciencia de Datos con una arquitectura preparada para evolucionar hacia un entorno productivo.

---

# 🎯 Objetivo

Construir un flujo reproducible que permita:

- Cargar y explorar datos financieros.
- Generar variables predictivas.
- Entrenar y comparar modelos de clasificación.
- Seleccionar el mejor modelo.
- Preparar el despliegue mediante Docker.
- Monitorear continuamente el desempeño del modelo.

---

# 🏗️ Arquitectura del Pipeline

```text
Carga de datos
       │
       ▼
Análisis Exploratorio (EDA)
       │
       ▼
Feature Engineering
       │
       ▼
Modelo Benchmark
       │
       ▼
Entrenamiento de Modelos
       │
       ▼
Evaluación
       │
       ▼
Despliegue (Docker)
       │
       ▼
Monitoreo
```

---

# 📁 Estructura del repositorio

```text
credit-scoring-mlops
│
├── src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── heuristic_model.py
│   ├── model_training.py
│   ├── model_deploy.py
│   ├── model_evaluation.py
│   └── model_monitoring.py
│
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
├── README.md
└── set_up.bat
```

---

# ⚙️ Componentes principales

| Archivo | Descripción |
|----------|-------------|
| `Cargar_datos.ipynb` | Carga inicial de los datos desde un archivo CSV de ejemplo. |
| `comprension_eda.ipynb` | Análisis exploratorio de datos y comprensión del problema. |
| `ft_engineering.py` | Ingeniería de variables y preparación del conjunto de datos para modelado. |
| `heuristic_model.py` | Implementación de un modelo heurístico utilizado como benchmark inicial. |
| `model_training.py` | Entrenamiento, evaluación y comparación de modelos de Machine Learning. |
| `model_deploy.py` | Preparación del modelo para su despliegue mediante Docker. |
| `model_evaluation.py` | Evaluación del modelo desplegado y generación de métricas. |
| `model_monitoring.py` | Monitoreo continuo para detectar degradación del desempeño y data drift. |

---

# 🐳 Ejecución del proyecto

### Construcción de la imagen Docker

```bash
docker build -t api-creditos .
```

### Ejecución del contenedor

```bash
docker run -d -p 8000:8000 --name api-creditos api-creditos
```

---

# 🛠️ Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Docker
- Git
- GitHub

---

# 💼 Competencias demostradas

- Machine Learning
- Clasificación supervisada
- Ingeniería de variables
- Evaluación y comparación de modelos
- Organización modular de proyectos
- Despliegue de modelos mediante Docker
- Monitoreo de modelos
- Buenas prácticas de MLOps
- Versionado con Git y GitHub

---

# ⭐ Valor agregado del proyecto

- Implementa un flujo completo de Machine Learning bajo principios de MLOps.
- Separa claramente las distintas etapas del ciclo de vida del modelo.
- Incluye un modelo benchmark para facilitar la comparación de desempeño.
- Incorpora despliegue y monitoreo, componentes fundamentales en proyectos productivos.
- Presenta una estructura modular, escalable y fácil de mantener.

---

## 👩‍💻 Autora

**Vanina Cavallin**

Doctora en Ciencias Biológicas | Jr. Data Scientist
