import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Riesgo Actuarial",
    layout="wide"
)

# ==========================================
# Nueva Paleta de Colores Elegante (Deep Navy / Emerald / Mint)
# ==========================================
st.markdown("""
<style>
    /* Fondo de la app */
    .stApp {
        background-color: #0B0F19;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Contenedor de Métricas */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #111827, #1F2937);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Valor de la Métrica */
    div[data-testid="stMetricValue"] {
        color: #10B981;
        font-weight: 700;
    }
    
    /* Botón Principal de la Barra Lateral */
    .stButton>button {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        color: white;
        transform: translateY(-1px);
    }
    
    /* Encabezados */
    h1, h2, h3, h4 {
        color: #F9FAFB;
        font-weight: 600;
    }
    
    /* Textos informativos secundarios */
    .stMarkdown p {
        color: #9CA3AF;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado Principal estilizado
st.title("🛡️ Sistema de Segmentación y Riesgo Actuarial")
st.caption("Inteligencia Artificial (K-means) | Allan Manuel Orellana - 20211920128")
st.markdown("Esta plataforma analiza los datos del asegurado en tiempo real para asignarle un segmento de salud y determinar el nivel de riesgo financiero asociado.")

# ==========================================
# Carga de Modelos y Datos
# ==========================================
@st.cache_resource
def cargar_recursos():
    ruta_kmeans = "models/kmeans_riesgo_actuarial.pkl"
    ruta_metadata = "models/model_metadata.json"
    ruta_csv = "insurance.csv"
    
    modelo_kmeans = joblib.load(ruta_kmeans) if os.path.exists(ruta_kmeans) else None
    
    metadata = {}
    if os.path.exists(ruta_metadata):
        with open(ruta_metadata, "r") as f:
            metadata = json.load(f)
            
    df_clean = pd.read_csv(ruta_csv) if os.path.exists(ruta_csv) else None
    
    return modelo_kmeans, metadata, df_clean

modelo, metadata, df = cargar_recursos()

if modelo is None:
    st.error("No se encontró el modelo K-means en `models/kmeans_riesgo_actuarial.pkl`. Por favor verifica la ruta.")
    st.stop()

# ==========================================
# Nueva Experiencia de Usuario: Formulario en la Barra Lateral
# ==========================================
st.sidebar.header("📋 Parámetros del Asegurado")

with st.sidebar.form("formulario_cliente"):
    age = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
    children = st.number_input("Número de hijos", min_value=0, max_value=10, value=1, step=1)
    
    st.markdown("---")
    sexo_opciones = {"Femenino": "female", "Masculino": "male"}
    sex_es = st.radio("Sexo", options=list(sexo_opciones.keys()), horizontal=True)
    sex = sexo_opciones[sex_es]

    fumador_opciones = {"Sí": "yes", "No": "no"}
    smoker_es = st.radio("¿Es Fumador?", options=list(fumador_opciones.keys()), horizontal=True)
    smoker = fumador_opciones[smoker_es]
    
    st.markdown("---")
    bmi = st.slider("Índice de Masa Corporal (BMI)", min_value=15.0, max_value=60.0, value=28.5, step=0.1)
    
    region_opciones = {
        "Suroeste (Southwest)": "southwest", 
        "Sureste (Southeast)": "southeast", 
        "Noroeste (Northwest)": "northwest", 
        "Noreste (Northeast)": "northeast"
    }
    region_es = st.selectbox("Región de Residencia", options=list(region_opciones.keys()))
    region = region_opciones[region_es]

    charges = st.slider("Cargos Médicos Anuales ($)", min_value=100.0, max_value=100000.0, value=13000.0, step=500.0)

    st.markdown("<br>", unsafe_allow_html=True)
    enviado = st.form_submit_button("⚡ Ejecutar Análisis Actuarial")

if not enviado:
    st.info("👈 Por favor, ingresa los datos del asegurado en el panel izquierdo y haz clic en 'Ejecutar Análisis Actuarial'.")
    st.stop()

# Diccionario para procesamiento interno (sin alterar)
cliente_dict = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
    "charges": charges
}
df_cliente = pd.DataFrame([cliente_dict])

# ==========================================
# Procesamiento y Predicción
# ==========================================
datos_prediccion = df_cliente.copy()

try:
    cluster_asignado = int(modelo.predict(datos_prediccion)[0])
except ValueError as e:
    columnas_ordenadas = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    datos_prediccion = datos_prediccion[columnas_ordenadas]
    cluster_asignado = int(modelo.predict(datos_prediccion)[0])

riesgo_mapeo = {0: "Bajo", 1: "Medio", 2: "Alto"}
explicacion_mapeo = {
    0: "Cluster con clientes jóvenes o adultos con hábitos saludables (no fumadores) y bajos cargos médicos.",
    1: "Cluster intermedio. Clientes con edad avanzada o BMI moderadamente alto, presentando reclamos médicos estándar.",
    2: "Cluster de Riesgo Crítico. Compuesto principalmente por clientes fumadores y/o con altos índices de masa corporal con cargos médicos sumamente elevados."
}

nivel_riesgo = riesgo_mapeo.get(cluster_asignado, "No definido")
explicacion_cluster = explicacion_mapeo.get(cluster_assigned=cluster_asignado, default="Segmento de clientes analizado por patrones de costos y comportamiento.") if hasattr(explicacion_mapeo, 'get') else explicacion_mapeo.get(cluster_asignado)

# ==========================================
# Sección Principal - Resultados Estilizados
# ==========================================
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 📊 Resultado del Diagnóstico")
    
    st.metric(label="Clasificación del Segmento", value=f"Cluster {cluster_asignado}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Notificaciones con nuevo diseño limpio nativo
    if nivel_riesgo == "Alto":
        st.error(f"🚨 **Nivel de Riesgo: {nivel_riesgo}**")
    elif nivel_riesgo == "Medio":
        st.warning(f"⚠️ **Nivel de Riesgo: {nivel_riesgo}**")
    else:
        st.success(f"✅ **Nivel de Riesgo: {nivel_riesgo}**")
        
    st.markdown(f"**Interpretación estadística:** {explicacion_cluster}")

with col2:
    st.markdown("### 📈 Posicionamiento en la Población")
    if df is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        # Ajuste de colores del gráfico al nuevo entorno oscuro
        fig.patch.set_facecolor("#0B0F19")
        ax.set_facecolor("#111827")

        # Nueva paleta del gráfico coordinada (Esmeralda para No Fumadores, Coral Suave para Fumadores)
        paleta_personalizada = {"yes": "#EF4444", "no": "#10B981"}

        sns.scatterplot(
            data=df, 
            x="bmi", 
            y="charges", 
            hue="smoker", 
            alpha=0.4, 
            palette=paleta_personalizada, 
            ax=ax
        )
        
        # Indicador del cliente con una estrella de alto contraste amarillo neón
        ax.scatter(bmi, charges, color="#FBBF24", s=250, marker="*", edgecolor="#FFFFFF", linewidth=1.5, label="Cliente Evaluado")
        
        # Limpieza visual de ejes y etiquetas
        ax.set_title("Matriz BMI vs Cargos Médicos Anuales", color="#F9FAFB", fontsize=11, pad=12)
        ax.set_xlabel("Índice de Masa Corporal (BMI)", color="#9CA3AF", fontsize=9)
        ax.set_ylabel("Cargos Médicos ($)", color="#9CA3AF", fontsize=9)
        ax.tick_params(colors="#9CA3AF", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.1)
        
        legend = ax.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#F9FAFB", loc="upper left", fontsize=8)
        
        st.pyplot(fig)
    else:
        st.info("Sube el archivo `insurance.csv` para habilitar el mapeo interactivo de dispersión.")

# ==========================================
# Tabla de Resumen
# ==========================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("### 📋 Resumen de la Ficha Evaluada")

df_bonito = pd.DataFrame([{
    "Edad": age,
    "Sexo": sex_es,       
    "BMI (IMC)": bmi,
    "Hijos": children,
    "Fumador": smoker_es,  
    "Región": region_es,     
    "Cargos Médicos": f"${charges:,.2f}"
}])

st.dataframe(df_bonito, use_container_width=True)
