# Predicción de Demanda con Machine Learning — Corporación Favorita 🛒

## ¿Cuánto le cuesta a un supermercado equivocarse en la predicción de demanda?

Análisis end-to-end de predicción de demanda para una cadena real de supermercados de Ecuador, con foco en la **traducción del error del modelo a costo de negocio real**.

---

## El problema de negocio

Predecir mal la demanda tiene dos tipos de error con costos distintos:

- **Understock** (predecir menos de lo que se vende) → venta perdida + cliente insatisfecho
- **Overstock** (predecir más de lo que se vende) → producto perecedero que se desperdicia

El "mejor modelo" no es el más preciso estadísticamente, sino el que **minimiza el costo total para el negocio**. Este proyecto demuestra esa diferencia con números reales.

---

## Dataset

**Store Sales - Time Series Forecasting — Corporación Favorita (Kaggle)**
- Cadena real de supermercados de Ecuador
- 3.000.888 registros de ventas diarias (enero 2013 — agosto 2017)
- 54 tiendas, 33 familias de producto
- Variables externas: precio del petróleo, feriados nacionales, promociones

**Foco del análisis:**
- 18 tiendas de Quito (ciudad principal)
- 6 familias que concentran el 80% de la facturación
- 108 series temporales × 4,6 años de historia

---

## Hallazgos del EDA

### Facturación por familia de producto
![Facturación por familia](https://raw.githubusercontent.com/lucasees/proyecto4-demanda-favorita/main/NOTEBOOKS/grafico1_ventas_familia.png)

GROCERY I domina con el **36,7%** de las ventas. BEVERAGES con **24,9%**. Juntos representan más del 60% del negocio.

### Evolución temporal de ventas
![Evolución temporal](https://raw.githubusercontent.com/lucasees/proyecto4-demanda-favorita/main/NOTEBOOKS/grafico2_evolucion_temporal.png)


Tendencia claramente **ascendente** de 2013 a 2017. La media móvil de 30 días llega a ~700.000 en 2017.

### Estacionalidad semanal
![Estacionalidad semanal](https://raw.githubusercontent.com/lucasees/proyecto4-demanda-favorita/main/NOTEBOOKS/grafico3_estacionalidad_semanal.png)

El **domingo es el día de mayor venta** (~3.447 promedio). El modelo debe capturar este patrón semanal.

### Impacto de las promociones
![Efecto promoción](https://raw.githubusercontent.com/lucasees/proyecto4-demanda-favorita/main/NOTEBOOKS/grafico4_efecto_promocion.png)

Las promociones **duplican las ventas** (+104% uplift). Sin promoción: 1.606 unidades promedio. Con promoción: 3.275.

### Ventas vs. precio del petróleo
![Petróleo](https://raw.githubusercontent.com/lucasees/proyecto4-demanda-favorita/main/NOTEBOOKS/grafico5_petroleo.png)

Relación **inversa**: cuando cae el precio del petróleo, suben las ventas. Ecuador es una economía petrolera — la caída del crudo lleva a medidas de austeridad que empujan el consumo local.

### Efecto de los feriados nacionales
![Feriados](https://raw.githubusercontent.com/lucasees/proyecto4-demanda-favorita/main/NOTEBOOKS/grafico6_feriados.png)

Los feriados nacionales tienen un impacto **prácticamente nulo** (+0,3%). Esto le dice al equipo de operaciones que no hace falta sobreabastecerse para feriados.

---

## Modelo — LightGBM

### Feature Engineering
13 variables predictoras construidas a partir del EDA:
- **Temporales:** día de semana, mes, año, día del mes
- **Lags:** ventas de los 7, 14 y 28 días anteriores
- **Rolling means:** media móvil de 7 y 28 días
- **Externas:** precio del petróleo, flag de promoción
- **Encoding:** familia de producto y tienda

### Importancia de variables
![Feature Importance](https://raw.githubusercontent.com/lucasees/proyecto4-demanda-favorita/main/NOTEBOOKS/grafico7_feature_importance.png)

Las variables temporales dominan. El **día del mes** es el predictor más importante, seguido del mes y la media móvil de 7 días.

### Métricas del modelo
| Métrica | Baseline | LightGBM | Mejora |
|---------|----------|----------|--------|
| MAE | 1.121 | 419 | **-62,6%** |
| RMSE | 2.047 | 907 | **-55,7%** |

---

## El diferencial: impacto económico
![Impacto económico](https://raw.githubusercontent.com/lucasees/proyecto4-demanda-favorita/main/NOTEBOOKS/grafico8_impacto_economico.png)

| | Baseline | Modelo LightGBM |
|---|---|---|
| Costo total (8 meses) | $40,4M | $12,8M |
| **Ahorro estimado** | | **$27,6M (68,3%)** |

El modelo no solo es más preciso — es **68,3% más barato** en términos de costo operativo real. En 8 meses de predicción para las tiendas de Quito, la diferencia entre usar el promedio histórico y este modelo equivale a **$27,6 millones en costos evitados**.

---

## Stack técnico

- **Python 3.11** — pandas, numpy, matplotlib, seaborn
- **LightGBM** — modelo de gradient boosting
- **scikit-learn** — métricas de evaluación
- **Jupyter Lab** — entorno de desarrollo
- **Dataset:** Kaggle — Store Sales Time Series Forecasting

---

## Estructura del proyecto


---

## Contacto

**Lucas Espinosa** — Data Analyst
[LinkedIn](https://linkedin.com/in/lucasespinosaa) · lucaseespinosa93@gmail.com · [github.com/lucasees](https://github.com/lucasees)
