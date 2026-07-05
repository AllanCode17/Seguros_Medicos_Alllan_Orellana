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
# Paleta de colores personalizada (Tonos morados / teal)
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #0f1420;
    }
    div[data-testid="stMetric"] {
        background-color: #1c2333;
        border: 1px solid #3a4a6b;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="stMetricValue"] {
        color: #7de3d1;
    }
    .stButton>button {
        background-color: #6c63ff;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #574fd6;
        color: white;
    }
    h1, h2, h3 {
        color: #e6e6fa;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.title("Sistema de Segmentación y Riesgo Actuarial (K-means) - Inteligencia Artificial -Allan Manuel Orellana - 20211920128")
st.markdown("""
Esta aplicación permite ingresar los datos de un asegurado para asignarlo a un segmento (Cluster) 
y determinar su nivel de riesgo actuarial.
""")

# ==========================================
# Carga de Modelos y Datos
# ==========================================
@st.cache_resource
def cargar_recursos():
    # Rutas relativas estándar
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

# Verificar que el modelo principal esté cargado
if modelo is None:
    st.error("No se encontró el modelo K-means en `models/kmeans_riesgo_actuarial.pkl`. Por favor verifica la ruta.")
    st.stop()

# ==========================================
# Entrada de Datos del Cliente - Formulario en el área principal (en Español)
# ==========================================
st.subheader("Datos del Cliente")

with st.form("formulario_cliente"):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        age = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
        children = st.number_input("Número de hijos", min_value=0, max_value=10, value=1, step=1)

    with col_b:
        sexo_opciones = {"Femenino": "female", "Masculino": "male"}
        sex_es = st.radio("Sexo", options=list(sexo_opciones.keys()), horizontal=True)
        sex = sexo_opciones[sex_es]

        fumador_opciones = {"Sí": "yes", "No": "no"}
        smoker_es = st.radio("¿Es Fumador?", options=list(fumador_opciones.keys()), horizontal=True)
        smoker = fumador_opciones[smoker_es]

    with col_c:
        bmi = st.slider("Índice de Masa Corporal (BMI)", min_value=15.0, max_value=60.0, value=28.5, step=0.1)
        region_opciones = {
            "Suroeste (Southwest)": "southwest", 
            "Sureste (Southeast)": "southeast", 
            "Noroeste (Northwest)": "northwest", 
            "Noreste (Northeast)": "northeast"
        }
        region_es = st.selectbox("Región", options=list(region_opciones.keys()))
        region = region_opciones[region_es]

    charges = st.slider("Cargos Médicos Anuales ($)", min_value=100.0, max_value=100000.0, value=13000.0, step=500.0)

    enviado = st.form_submit_button("Calcular Segmento de Riesgo")

if not enviado:
    st.info("Completa el formulario y presiona 'Calcular Segmento de Riesgo' para ver los resultados.")
    st.stop()

# El diccionario interno que va al modelo SE MANTIENE en inglés para que no falle el pipeline
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
# El Pipeline espera TODAS las columnas del cliente en el orden o formato correcto
datos_prediccion = df_cliente.copy()

# Realizar la predicción pasando el dataframe completo del cliente
try:
    cluster_asignado = int(modelo.predict(datos_prediccion)[0])
except ValueError as e:
    # En caso de que se necesite validar un orden específico de columnas
    columnas_ordenadas = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    datos_prediccion = datos_prediccion[columnas_ordenadas]
    cluster_asignado = int(modelo.predict(datos_prediccion)[0])

# Mapear el nivel de riesgo en base a los criterios de tu notebook o metadata
# Si tu metadata.json define los riesgos, los extraemos de ahí de forma dinámica
riesgo_mapeo = {0: "Bajo", 1: "Medio", 2: "Alto"} # <- Ajusta los números según los resultados de tu gráfico
explicacion_mapeo = {
    0: "Cluster con clientes jóvenes o adultos con hábitos saludables (no fumadores) y bajos cargos médicos.",
    1: "Cluster intermedio. Clientes con edad avanzada o BMI moderadamente alto, presentando reclamos médicos estándar.",
    2: "Cluster de Riesgo Crítico. Compuesto principalmente por clientes fumadores y/o con altos índices de masa corporal con cargos médicos sumamente elevados."
}

nivel_riesgo = riesgo_mapeo.get(cluster_asignado, "No definido")
explicacion_cluster = explicacion_mapeo.get(cluster_asignado, "Segmento de clientes analizado por patrones de costos y comportamiento.")

# ==========================================
# Sección Principal - Resultados
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Resultado de Segmentación")
    
    # Contenedor visual para el Cluster y Riesgo
    st.metric(label="Cluster Asignado", value=f"Cluster {cluster_asignado}")
    
    # Color dinámico según el riesgo
    if nivel_riesgo == "Alto":
        st.error(f"Riesgo Actuarial: {nivel_riesgo}")
    elif nivel_riesgo == "Medio":
        st.warning(f"Riesgo Actuarial: {nivel_riesgo}")
    else:
        st.success(f"Riesgo Actuarial: {nivel_riesgo}")
        
    st.write(f"Análisis del Perfil: {explicacion_cluster}")

with col2:
    st.subheader("Contexto General del Modelo")
    if df is not None:
        # Gráfico de dispersión interactivo/visual que muestra dónde se ubica el cliente actual
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#0f1420")
        ax.set_facecolor("#1c2333")

        # Paleta personalizada morado/teal en lugar de "Set2"
        paleta_personalizada = {"yes": "#ff6b81", "no": "#7de3d1"}

        sns.scatterplot(
            data=df, 
            x="bmi", 
            y="charges", 
            hue="smoker", 
            alpha=0.6, 
            palette=paleta_personalizada, 
            ax=ax
        )
        
        # Destacar la posición del cliente ingresado
        ax.scatter(bmi, charges, color="#ffd166", s=220, marker="*", edgecolor="white", linewidth=1, label="Cliente Actual")
        ax.set_title("Distribución de Clientes: BMI vs Cargos Médicos", color="#e6e6fa")
        ax.set_xlabel("Índice de Masa Corporal (BMI)", color="#e6e6fa")
        ax.set_ylabel("Cargos Médicos ($)", color="#e6e6fa")
        ax.tick_params(colors="#e6e6fa")
        ax.legend(facecolor="#1c2333", labelcolor="#e6e6fa")
        
        st.pyplot(fig)
    else:
        st.info("Sube el archivo `insurance.csv` para habilitar las visualizaciones estadísticas de los clusters.")

# ==========================================
# Detalles del Cliente Ingresado (Traducido al Español)
# ==========================================
st.markdown("---")
st.subheader("Resumen de datos enviados")

# Creamos una copia en español para mostrarla en la interfaz
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
