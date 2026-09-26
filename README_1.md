# Random Forest · Tensión de liquidez

App de Streamlit que entrena un modelo **Random Forest de clasificación** para predecir si una empresa presenta **tensión de liquidez** (`Tension_Liquidez_bin`: 1 = tensión, 0 = normal) a partir de variables de capital de trabajo.

> Modelo básico con fines didácticos. La base de ejemplo es 100% simulada y no representa empresas reales.

## Archivos del repositorio

| Archivo | Descripción |
|---|---|
| `app_rf_tension_liquidez.py` | Aplicación de Streamlit (entrenamiento, evaluación y predicción) |
| `requirements.txt` | Librerías necesarias |
| `README.md` | Este documento |

## Instalación y ejecución local

Requiere **Python 3.10 o superior**.

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
pip install -r requirements.txt
streamlit run app_rf_tension_liquidez.py
```

La app abre en el navegador (`http://localhost:8501`). Desde el panel lateral se sube el Excel de entrenamiento.

## Publicar en Streamlit Community Cloud

1. Sube los 3 archivos a un repositorio de GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io) → **Create app**.
3. Selecciona el repositorio, la rama y como *Main file path* `app_rf_tension_liquidez.py`.
4. **Deploy**. La app instala automáticamente lo que indica `requirements.txt`.

## Datos de entrada

Excel (`.xlsx` / `.xlsm`) con la hoja **`Datos_Modelo`**. Si no existe esa hoja, se lee la primera.

| Tipo | Variables |
|---|---|
| **Target** | `Tension_Liquidez_bin` |
| **Predictores (21)** | `Sector`, `Ventas_12M_MXN`, `Crecimiento_Ventas_pct`, `Margen_EBITDA_pct`, `DSO_dias`, `DIO_dias`, `DPO_dias`, `CCC_dias`, `CxC_MXN`, `Inventarios_MXN`, `CxP_MXN`, `NWC_MXN`, `Deuda_CP_MXN`, `Caja_MXN`, `Linea_Credito_Disponible_MXN`, `Concentracion_Top5_Clientes_pct`, `Morosidad_CxC_pct`, `Inventario_Obsoleto_pct`, `Volatilidad_Ventas_pct`, `Estacionalidad_bin`, `EBITDA_MXN` |
| **Excluidas** | `ID_Observacion`, `Fecha`, `Prob_Tension_Liquidez`, `Brecha_Caja_90d_MXN` |

**Razón de las exclusiones:**

- `ID_Observacion`: consecutivo sin contenido económico.
- `Fecha`: la base es simulada; usarla introduce tendencia artificial.
- `Prob_Tension_Liquidez`: probabilidad con la que se generó la etiqueta → fuga de información.
- `Brecha_Caja_90d_MXN`: resultado de los siguientes 90 días (dato futuro) y target del modelo de regresión.

## Configuración del modelo

| Componente | Configuración |
|---|---|
| Algoritmo | `RandomForestClassifier` (scikit-learn) |
| Codificación | `Sector` con `OneHotEncoder(handle_unknown="ignore")`; numéricas sin transformar |
| Partición | 70% entrenamiento / 30% prueba, aleatoria y estratificada, `random_state=42` |
| Balance de clases | `class_weight="balanced"` |
| Hiperparámetros (default) | `n_estimators=300`, `max_depth=8`, `min_samples_leaf=1` (ajustables con sliders) |
| Umbral de decisión | 0.50 por defecto, ajustable de 0.05 a 0.95 |
| Métricas | Accuracy, Precision, Recall, F1 y ROC-AUC |
| Reporte | Excel descargable con lo revisado e interpretación en lenguaje sencillo |

## Secciones de la app

1. **Exploración**: vista previa, distribución del target, tasa de tensión por sector y estadísticos descriptivos.
2. **Desempeño**: métricas de prueba vs entrenamiento, matriz de confusión y reporte de clasificación.
3. **Curva ROC**: prueba y entrenamiento, marcando el umbral seleccionado.
4. **Importancia de variables**: `feature_importances_` del Random Forest.
5. **Predicción nuevas empresas**: descarga de plantilla, carga de un Excel con empresas nuevas y descarga de resultados (`Prob_Tension_Modelo`, `Tension_Predicha`, `Clasificacion`, `Umbral_usado`).
6. **Reporte**: botón *Descargar reporte (.xlsx)* con la configuración y el umbral vigentes al momento de descargar.

## Reporte descargable (.xlsx)

Cada hoja incluye una columna **Interpretación** en lenguaje sencillo.

| Hoja | Contenido |
|---|---|
| Resumen | Fecha, archivo, configuración usada y conclusiones clave generadas automáticamente (AUC, recall, tensiones no detectadas, falsas alarmas, variable principal, brecha entrenamiento-prueba) |
| Métricas | Accuracy, Precision, Recall, F1 y ROC-AUC en prueba vs entrenamiento, con la diferencia y su lectura (brecha baja ≤ 0.05, moderada ≤ 0.10, alta > 0.10) |
| Matriz | Aciertos, falsas alarmas y tensiones no detectadas, con número y % |
| Umbrales | Umbrales de 0.10 a 0.90 (pasos de 0.05): tensión detectada / no detectada, falsas alarmas y métricas; el umbral vigente va resaltado |
| Importancia | Las 21 variables ordenadas (Sector agregado), peso %, acumulado y descripción |
| Predicciones | Solo si se subió archivo de empresas nuevas: probabilidad, clasificación y **Lectura** |

**Lectura de predicciones** (se mueve con el umbral; ancho de ±0.10):

- **Alerta clara**: probabilidad ≥ umbral + 0.10
- **Cerca del umbral: revisar**: entre umbral − 0.10 y umbral + 0.10
- **Sin alerta**: probabilidad < umbral − 0.10

## Resultados de referencia

Con la base didáctica (600 obs.) y la configuración por defecto, sobre el conjunto de prueba (180 obs., umbral 0.50):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.844 | 0.843 | 0.892 | 0.867 | 0.918 |

## Consideraciones

- `CCC_dias` = DSO + DIO − DPO y `NWC_MXN` = CxC + Inventarios − CxP son combinaciones exactas; la importancia se reparte entre variables redundantes.
- Los montos están en MXN absolutos, por lo que el modelo también capta el tamaño de la empresa.
- Mejoras posibles: razones financieras (Caja/Deuda CP, NWC/Ventas, Deuda CP/EBITDA), validación cruzada y búsqueda de hiperparámetros (`GridSearchCV`).
