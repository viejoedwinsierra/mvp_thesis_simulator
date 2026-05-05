# Proyecto: Simulación y Modelamiento Estadístico para Blob Storage

## Descripción General

Este proyecto implementa un pipeline completo basado en simulación Monte Carlo para analizar, modelar y evaluar el comportamiento de un sistema tipo Blob Storage.

El flujo cubre:

- Generación de datos simulados
- Análisis descriptivo y multivariado
- Pre-modelamiento
- Modelamiento estadístico
- Evaluación de modelos sin reentrenamiento

El objetivo es construir un sistema interpretable que permita entender y evaluar desempeño, costo y fallas.

---

## Arquitectura del Pipeline

```text
run_simulation.py
    ↓
run_analysis.py
    ↓
run_premodeling_modeling_.py
    ↓
evaluación de modelos (opcional)
```

---

## 1. Simulación

```bash
python src/run_simulation.py
```

Salida:

```text
output/datasets/
```

---

## 2. Análisis

```bash
python src/run_analysis.py
```

Salida:

```text
output/html/report_analysis.html
output/plots/
```

---

## 3. Pre-modeling

Prepara datos para modelamiento:

- selección de variables
- eliminación de leakage
- encoding
- transformaciones

Salida:

```text
output/html/report_premodeling.html
output/premodeling/<target>/
```

---

## 4. Modeling

Entrena modelos estadísticos:

- OLS log-lineal
- GLM Gamma
- Regresión logística

```bash
python src/run_premodeling_modeling_.py
```

Salida:

```text
output/html/report_modeling.html
output/plots/modeling/
```

---

## 5. Evaluación de modelos

### Entrenamiento

```bash
python src/run_train_statistical_models.py
```

Salida:

```text
output/models/statistical/
```

### Nuevos datos

```bash
python src/run_simulation.py
```

### Evaluación

```bash
python src/run_evaluate_saved_models.py
```

Salida:

```text
output/html/report_model_evaluation.html
output/model_evaluation/
output/plots/model_evaluation/
```

---

## Diferencias clave

| Fase | Función |
|------|--------|
| Modeling | entrena + evalúa |
| Train | entrena y guarda |
| Evaluation | evalúa sin reentrenar |

---

## Dependencias

```text
pandas
numpy
matplotlib
scikit-learn
statsmodels
```

---

## Enfoque

- Modelos interpretables
- Sin ML complejo
- Orientado a análisis

---

## Uso recomendado

### Presentación

```bash
python src/run_premodeling_modeling_.py
```

### Validación completa

```bash
python src/run_train_statistical_models.py
python src/run_simulation.py
python src/run_evaluate_saved_models.py
```

---

## Nota

Este pipeline separa claramente:

```text
entrenar ≠ evaluar
```
