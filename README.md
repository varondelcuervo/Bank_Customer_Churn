# 🏦 Bank Customer Churn Prediction

Proyecto de Machine Learning para la predicción de abandono de clientes bancarios (*Customer Churn*). El proyecto incluye el ciclo completo de un modelo analítico: análisis exploratorio, ingeniería de variables, entrenamiento y comparación de modelos, interpretación, evaluación, monitoreo y despliegue mediante una API REST utilizando **FastAPI**.

---

## 🚀 Objetivos

- Analizar los factores asociados al abandono de clientes.
- Construir un modelo predictivo para estimar la probabilidad de churn.
- Comparar diferentes algoritmos de Machine Learning.
- Explicar las predicciones mediante técnicas de interpretabilidad.
- Exponer el modelo mediante una API REST lista para producción.

---

## 📂 Estructura del proyecto

```text
Bank-Churn/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # API FastAPI
│   ├── predictor.py         # Carga de modelos y predicciones
│   ├── schemas.py           # Modelos Pydantic
│   └── transformers.py      # Feature Engineering personalizado
│
├── datasets/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── models/
│   ├── logistic_pipeline.pkl
│   └── gradient_pipeline.pkl
│
├── Bank_Customer_Churn.ipynb
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

El proyecto utiliza el dataset **Bank Customer Churn**, cuyo objetivo es predecir si un cliente abandonará el banco (`Exited`).

Variable objetivo:

- **Exited**
    - 1 → Cliente abandona el banco
    - 0 → Cliente permanece

Variables utilizadas:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

---

# ⚙️ Ingeniería de Variables

Se desarrolló un transformador personalizado compatible con Scikit-Learn para generar nuevas variables automáticamente dentro del Pipeline.

Variables creadas:

| Variable | Descripción |
|-----------|-------------|
| BalancePerProduct | Balance promedio por producto |
| AgeGroup | Grupo de edad |
| HasBalance | Indicador de balance positivo |
| EngagedCustomer | Cliente activo con dos o más productos |
| TenureAgeRatio | Relación entre antigüedad y edad |
| BalanceSalaryRatio | Relación entre balance y salario |

Todo el Feature Engineering se ejecuta automáticamente durante entrenamiento e inferencia.

---

# 🤖 Modelos entrenados

Se evaluaron diferentes algoritmos de clasificación:

- Logistic Regression
- Random Forest
- Gradient Boosting

La selección del mejor modelo se realizó mediante:

- GridSearchCV
- Cross Validation
- Optimización de hiperparámetros

---

# 📈 Evaluación

Se calcularon diferentes métricas de clasificación:

- Accuracy
- Precision
- Recall
- F1-Score

Adicionalmente se realizaron análisis mediante:

- Matriz de confusión
- ROC Curve
- Feature Importance
- SHAP Values
- Kolmogorov-Smirnov (KS)
- Population Stability Index (PSI)

---

# 🔍 Interpretabilidad

Para explicar las predicciones del modelo se utilizaron:

- Coeficientes y Odds Ratio (Logistic Regression)
- Feature Importance
- SHAP Summary Plot
- SHAP Feature Importance

Esto permite comprender cuáles variables tienen mayor impacto sobre la probabilidad de abandono.

---

# 🌐 API REST

Los modelos fueron serializados mediante Joblib y expuestos utilizando FastAPI.

## Ejecutar la API

```bash
uvicorn app.main:app --reload
```

La documentación interactiva estará disponible en:

```
http://127.0.0.1:8000/docs
```

---

## Endpoint

### POST

```
/predict/logistic
```

### POST

```
/predict/gradient
```

---

## Ejemplo de Request

```json
{
    "CreditScore":650,
    "Geography":"Germany",
    "Gender":"Male",
    "Age":45,
    "Tenure":5,
    "Balance":120000,
    "NumOfProducts":2,
    "HasCrCard":1,
    "IsActiveMember":1,
    "EstimatedSalary":85000
}
```

---

## Ejemplo de Response

```json
{
    "model":"Gradient Boosting",
    "prediction":1,
    "probability":0.8731
}
```

---

# 🧠 Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- SHAP
- FastAPI
- Joblib
- Pydantic

---

# 📌 Pipeline de Machine Learning

El proyecto utiliza un Pipeline completo de Scikit-Learn compuesto por:

```
Feature Engineering
        ↓
Preprocessing
        ↓
One Hot Encoding
        ↓
Standard Scaling
        ↓
Modelo
```

Esto garantiza que durante entrenamiento e inferencia se apliquen exactamente las mismas transformaciones.

---


---

# 📈 Preguntas de negocio a responder

- **Geografía:** cliente en Alemania (**37.9% de churn vs. 16-17% en Francia/España**) — el patrón geográfico más fuerte del dataset.

- **Número de productos:** tiene **1 producto (34.7% churn)** o, de forma extrema, **3-4 productos (87-88% churn)**. El punto óptimo es **2 productos (6.0% churn)**.

- **Actividad:** es un miembro inactivo (**IsActiveMember=0 → 29.7% churn vs. 12.5% en activos**).

- **Género:** es mujer (**28.0% churn vs. 15.9% en hombres**).

- **Edad:** clientes de mayor edad (**odds ratio +2.19x por Age estandarizada**); los grupos **41-50** y **51-60 años** tienen mayor riesgo, mientras que **60+** es marcadamente protector (posiblemente por lealtad de largo plazo o menor cambio de producto financiero a esa edad).

- **Balance:** balances más altos se asocian a mayor riesgo (**odds ratio +2.18x**) — contraintuitivo, pero consistente con clientes de mayor valor que tienen más opciones y son más cortejados por la competencia.

- **Engagement combinado:** la variable derivada **EngagedCustomer** (activo + 2+ productos) es fuertemente protectora (**odds ratio 0.44x**), lo que confirma que la combinación de actividad y vinculación de productos es más predictiva que cualquier variable aislada.

# Muestras de funcionamiento de la API

<img width="1078" height="947" alt="image" src="https://github.com/user-attachments/assets/07185818-6ddf-4439-9e82-951968731158" />

<img width="1072" height="924" alt="image" src="https://github.com/user-attachments/assets/8fbb6875-e76e-4862-a5eb-642daa0851d5" />

