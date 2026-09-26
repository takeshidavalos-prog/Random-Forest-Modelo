"""
Random Forest - Clasificación de Tensión de Liquidez (modelo básico)
Ejecutar:  streamlit run app_rf_tension_liquidez.py

Configuración acordada:
  1. Excluir: ID_Observacion, Fecha, Prob_Tension_Liquidez, Brecha_Caja_90d_MXN (evitar fuga)
  2. Predictores: 21 variables de la base tal cual; Sector con OneHotEncoder
  3. Partición 70/30 aleatoria estratificada, random_state=42, sin CV
  4. class_weight='balanced'
  5. Hiperparámetros de la guía (300 árboles, max_depth=8) ajustables con sliders
  6. Métricas: accuracy, precision, recall, F1, ROC-AUC; umbral ajustable (0.50 por defecto)
  7. Salidas: exploración, métricas + matriz de confusión, curva ROC, importancia de variables
  8. Carga de Excel desde la app + predicción en lote con descarga a Excel
"""

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------------
# Parámetros fijos del modelo
# ------------------------------------------------------------------
HOJA = "Datos_Modelo"
TARGET = "Tension_Liquidez_bin"
EXCLUIR = ["ID_Observacion", "Fecha", "Prob_Tension_Liquidez", "Brecha_Caja_90d_MXN"]
CATEGORICAS = ["Sector"]
TEST_SIZE = 0.30
RANDOM_STATE = 42

st.set_page_config(page_title="RF Tensión de Liquidez", layout="wide")
st.title("Random Forest · Tensión de liquidez")
st.caption("Clasificación binaria: 1 = tensión de liquidez, 0 = situación normal. Modelo básico.")


# ------------------------------------------------------------------
# Funciones
# ------------------------------------------------------------------
@st.cache_data
def leer_excel(contenido: bytes, hoja: str | None = HOJA) -> pd.DataFrame:
    xls = pd.ExcelFile(BytesIO(contenido))
    if hoja and hoja in xls.sheet_names:
        return xls.parse(hoja)
    return xls.parse(xls.sheet_names[0])


def columnas_predictoras(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUIR + [TARGET]]


@st.cache_resource
def entrenar(df: pd.DataFrame, n_estimators: int, max_depth: int, min_samples_leaf: int):
    predictores = columnas_predictoras(df)
    X = df[predictores]
    y = df[TARGET].astype(int)
    cat = [c for c in CATEGORICAS if c in X.columns]

    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", prep), ("model", modelo)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    proba_train = pipe.predict_proba(X_train)[:, 1]
    return pipe, predictores, X_train, X_test, y_train, y_test, proba_train, proba_test


def metricas(y_true, proba, umbral):
    pred = (proba >= umbral).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
    }, pred


def a_excel(df: pd.DataFrame, hoja: str) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=hoja, index=False)
    return buffer.getvalue()


# ------------------------------------------------------------------
# Barra lateral: datos e hiperparámetros
# ------------------------------------------------------------------
st.sidebar.header("1. Datos")
archivo = st.sidebar.file_uploader("Base de entrenamiento (.xlsx)", type=["xlsx", "xlsm"])

st.sidebar.header("2. Hiperparámetros")
n_estimators = st.sidebar.slider("Número de árboles (n_estimators)", 50, 1000, 300, 50)
max_depth = st.sidebar.slider("Profundidad máxima (max_depth)", 2, 20, 8, 1)
min_samples_leaf = st.sidebar.slider("Mínimo de obs. por hoja (min_samples_leaf)", 1, 20, 1, 1)

st.sidebar.header("3. Umbral de decisión")
umbral = st.sidebar.slider("Clasificar como tensión si probabilidad ≥", 0.05, 0.95, 0.50, 0.01)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Partición {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} estratificada · "
    f"random_state={RANDOM_STATE} · class_weight='balanced'"
)

if archivo is None:
    st.info("Sube el archivo Excel con la hoja **Datos_Modelo** para entrenar el modelo.")
    st.stop()

df = leer_excel(archivo.getvalue())

faltantes = [c for c in [TARGET] + CATEGORICAS if c not in df.columns]
if faltantes:
    st.error(f"No se encontraron las columnas requeridas: {', '.join(faltantes)}")
    st.stop()

predictores = columnas_predictoras(df)
if df[predictores + [TARGET]].isna().any().any():
    st.warning("La base tiene valores faltantes; se eliminan esas filas para entrenar.")
    df = df.dropna(subset=predictores + [TARGET])

with st.spinner("Entrenando Random Forest..."):
    pipe, predictores, X_train, X_test, y_train, y_test, proba_train, proba_test = entrenar(
        df, n_estimators, max_depth, min_samples_leaf
    )

m_test, pred_test = metricas(y_test, proba_test, umbral)
m_train, _ = metricas(y_train, proba_train, umbral)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Exploración", "Desempeño", "Curva ROC", "Importancia de variables", "Predicción nuevas empresas"]
)

# ------------------------------------------------------------------
# 1. Exploración
# ------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observaciones", f"{len(df):,}")
    c2.metric("Predictores", len(predictores))
    c3.metric("% con tensión (1)", f"{df[TARGET].mean():.1%}")
    c4.metric("Sectores", df["Sector"].nunique())

    st.markdown(f"**Variables excluidas:** {', '.join([c for c in EXCLUIR if c in df.columns])}")
    st.subheader("Vista previa")
    st.dataframe(df.head(20))

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Distribución del target")
        conteo = df[TARGET].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(["0 · Normal", "1 · Tensión"], conteo.values, color=["#4C78A8", "#E45756"])
        for i, v in enumerate(conteo.values):
            ax.text(i, v, f"{v} ({v/conteo.sum():.0%})", ha="center", va="bottom")
        ax.set_ylabel("Observaciones")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
    with g2:
        st.subheader("Tasa de tensión por sector")
        por_sector = (
            df.groupby("Sector")[TARGET].agg(Observaciones="count", Tasa_tension="mean")
            .sort_values("Tasa_tension")
        )
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(por_sector.index, por_sector["Tasa_tension"], color="#E45756")
        for i, v in enumerate(por_sector["Tasa_tension"]):
            ax.text(v, i, f" {v:.0%}", va="center")
        ax.set_xlim(0, 1)
        ax.set_xlabel("% con tensión de liquidez")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    st.subheader("Estadísticos descriptivos de los predictores numéricos")
    st.dataframe(df[predictores].describe().T.round(3))

# ------------------------------------------------------------------
# 2. Desempeño
# ------------------------------------------------------------------
with tab2:
    st.subheader(f"Métricas en el conjunto de prueba (umbral = {umbral:.2f})")
    cols = st.columns(5)
    for col, (nombre, valor) in zip(cols, m_test.items()):
        col.metric(nombre, f"{valor:.3f}", delta=f"{valor - m_train[nombre]:+.3f} vs entrenamiento",
                   delta_color="off")
    st.caption(
        f"Entrenamiento: {len(y_train)} obs. · Prueba: {len(y_test)} obs. "
        "Una diferencia grande entre entrenamiento y prueba indica sobreajuste."
    )

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Matriz de confusión")
        cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["0 · Normal", "1 · Tensión"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)
        tn, fp, fn, tp = cm.ravel()
        st.markdown(
            f"- **Falsos negativos (tensión no detectada):** {fn}\n"
            f"- **Falsos positivos (falsa alarma):** {fp}"
        )
    with g2:
        st.subheader("Reporte de clasificación")
        reporte = pd.DataFrame(
            classification_report(
                y_test, pred_test, target_names=["0 · Normal", "1 · Tensión"],
                output_dict=True, zero_division=0,
            )
        ).T.round(3)
        st.dataframe(reporte)

# ------------------------------------------------------------------
# 3. Curva ROC
# ------------------------------------------------------------------
with tab3:
    st.subheader(f"Curva ROC · AUC prueba = {m_test['ROC-AUC']:.3f}")
    fpr, tpr, thr = roc_curve(y_test, proba_test)
    fpr_tr, tpr_tr, _ = roc_curve(y_train, proba_train)
    idx = np.argmin(np.abs(thr - umbral))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E45756", lw=2, label=f"Prueba (AUC {m_test['ROC-AUC']:.3f})")
    ax.plot(fpr_tr, tpr_tr, color="#4C78A8", lw=1, ls="--",
            label=f"Entrenamiento (AUC {m_train['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls=":", label="Azar")
    ax.scatter(fpr[idx], tpr[idx], color="black", zorder=5, label=f"Umbral {umbral:.2f}")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

# ------------------------------------------------------------------
# 4. Importancia de variables
# ------------------------------------------------------------------
with tab4:
    st.subheader("Importancia de variables (feature_importances_)")
    nombres = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.DataFrame(
        {"Variable": nombres, "Importancia": pipe.named_steps["model"].feature_importances_}
    ).sort_values("Importancia", ascending=False).reset_index(drop=True)

    top_n = st.slider("Número de variables a mostrar", 5, len(imp), min(15, len(imp)))
    top = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * top_n + 1))
    ax.barh(top["Variable"], top["Importancia"], color="#4C78A8")
    ax.set_xlabel("Importancia (reducción media de impureza)")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    st.caption(
        "Nota: CCC_dias es combinación exacta de DSO, DIO y DPO, y NWC_MXN de CxC, Inventarios y CxP; "
        "la importancia se reparte entre variables redundantes."
    )
    st.dataframe(imp.assign(Importancia=imp["Importancia"].round(4)))

# ------------------------------------------------------------------
# 5. Predicción en lote
# ------------------------------------------------------------------
with tab5:
    st.subheader("Predicción para empresas nuevas")
    st.markdown(
        f"Sube un Excel con las mismas columnas predictoras ({len(predictores)}). "
        "Puede incluir columnas adicionales (ID, nombre, etc.); se conservan en el resultado."
    )
    plantilla = pd.DataFrame(columns=["ID_Empresa"] + predictores)
    st.download_button(
        "Descargar plantilla (.xlsx)",
        data=a_excel(plantilla, "Nuevas_Empresas"),
        file_name="plantilla_nuevas_empresas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    nuevo = st.file_uploader("Archivo de empresas nuevas (.xlsx)", type=["xlsx", "xlsm"], key="nuevo")
    if nuevo is not None:
        df_nuevo = leer_excel(nuevo.getvalue(), hoja=None)
        faltan = [c for c in predictores if c not in df_nuevo.columns]
        if faltan:
            st.error(f"Faltan columnas en el archivo: {', '.join(faltan)}")
        elif df_nuevo[predictores].isna().any().any():
            st.error("El archivo tiene valores vacíos en columnas predictoras; complétalos antes de predecir.")
        else:
            sectores_nuevos = set(df_nuevo["Sector"]) - set(df["Sector"])
            if sectores_nuevos:
                st.warning(f"Sectores no vistos en entrenamiento (se tratan como 'ninguno'): {sectores_nuevos}")
            proba = pipe.predict_proba(df_nuevo[predictores])[:, 1]
            resultado = df_nuevo.copy()
            resultado["Prob_Tension_Modelo"] = proba.round(4)
            resultado["Tension_Predicha"] = (proba >= umbral).astype(int)
            resultado["Clasificacion"] = np.where(resultado["Tension_Predicha"] == 1, "Tensión", "Normal")
            resultado["Umbral_usado"] = umbral

            c1, c2 = st.columns(2)
            c1.metric("Empresas evaluadas", len(resultado))
            c2.metric("Con tensión predicha", f"{resultado['Tension_Predicha'].sum()} "
                      f"({resultado['Tension_Predicha'].mean():.0%})")
            st.dataframe(resultado)
            st.download_button(
                "Descargar predicciones (.xlsx)",
                data=a_excel(resultado, "Predicciones"),
                file_name="predicciones_tension_liquidez.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
