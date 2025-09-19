import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Regresión Lineal - Estudiantes",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# CSS simplificado y optimizado
st.markdown("""
<style>
    .main {
        background-color: #fafbfe;
    }

    .main-header {
        background: linear-gradient(135deg, #4A6FDC 0%, #6C63FF 100%);
        border-radius: 12px;
        color: white;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border-top: 4px solid #4A6FDC;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .feature-box {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #6C63FF;
    }

    .navigation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .navigation-card:hover {
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>📊 Sistema Predictivo de Rendimiento Académico</h1>
    <p>Modelo de Regresión Lineal para predecir promedios estudiantiles</p>
</div>
""", unsafe_allow_html=True)

# Información del dataset
col1, col2, col3, col4 = st.columns(4)

if os.path.exists("dataset.xlsx"):
    try:
        df = pd.read_excel("dataset.xlsx")
        # Manejo robusto de la columna promedio
        if df["promedio"].dtype == 'object':
            # Si es string, reemplazar comas por puntos
            df["promedio"] = df["promedio"].str.replace(",", ".").astype(float)
        else:
            # Si ya es numérico, asegurar que sea float
            df["promedio"] = df["promedio"].astype(float)

        # Conversión explícita de tipos para evitar errores de PyArrow
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'promedio':
                # Intentar convertir a numérico si es posible, excepto promedio que se maneja aparte
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    # Si no se puede convertir, mantener como string
                    df[col] = df[col].astype(str)
            elif str(df[col].dtype).startswith('Int') or 'int64' in str(df[col].dtype):
                # Convertir enteros nullable a int64 estándar
                try:
                    df[col] = df[col].fillna(0).astype('int64')
                except:
                    df[col] = df[col].astype('object')

        df_clean = df.drop(columns=["tipo_documento", "documento", "nombre_completo"], errors='ignore')

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👥 Total Estudiantes</h3>
                <h2>{len(df)}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Variables</h3>
                <h2>{len(df_clean.columns)}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Promedio</h3>
                <h2>{df["promedio"].mean():.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>✅ Estado</h3>
                <h2>Cargado</h2>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.warning("⚠️ Error al cargar el dataset.")
        st.error(f"Detalles: {str(e)}")
else:
    st.warning("⚠️ Dataset no encontrado. Asegúrate de que 'dataset.xlsx' esté en el directorio.")

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["📋 Resumen", "🔬 Variables", "🧭 Navegación"])

with tab1:
    st.markdown("""
    <div class="feature-box">
        <h2>🎯 Objetivo del Proyecto</h2>
        <p>Sistema de <strong>regresión lineal múltiple</strong> para predecir el promedio académico
        de estudiantes basándose en factores clave del rendimiento educativo.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📊 Metodología</h3>
            <ul>
                <li><strong>Análisis Exploratorio:</strong> Comprensión de datos y relaciones</li>
                <li><strong>Preprocesamiento:</strong> Limpieza y preparación de variables</li>
                <li><strong>Modelado:</strong> Regresión lineal múltiple optimizada</li>
                <li><strong>Evaluación:</strong> Validación con métricas de rendimiento</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🎯 Beneficios</h3>
            <ul>
                <li><strong>Predicción temprana:</strong> Identificación de estudiantes en riesgo</li>
                <li><strong>Intervención dirigida:</strong> Estrategias personalizadas</li>
                <li><strong>Optimización educativa:</strong> Mejora del rendimiento académico</li>
                <li><strong>Toma de decisiones:</strong> Basada en datos y evidencia</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="feature-box">
        <h3>📋 Variables del Modelo</h3>
        <p>Factores clave que influyen en el rendimiento académico:</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="navigation-card">
            <h3>📚 Variables Académicas</h3>
            <ul>
                <li>Horas de estudio</li>
                <li>Asistencia a clase</li>
                <li>Participación en clase</li>
                <li>Tareas completadas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="navigation-card">
            <h3>👤 Variables Personales</h3>
            <ul>
                <li>Horas de sueño</li>
                <li>Estrato socioeconómico</li>
                <li>Actividad física</li>
                <li>Calidad de alimentación</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="navigation-card">
            <h3>🏠 Variables Contextuales</h3>
            <ul>
                <li>Acceso a tecnología</li>
                <li>Entorno de estudio</li>
                <li>Apoyo familiar</li>
                <li>Tiempo de transporte</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="feature-box">
        <h3>🧭 Navegación del Sistema</h3>
        <p>Explora las diferentes secciones del sistema predictivo:</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="navigation-card">
            <h3>1️⃣ Análisis Exploratorio (EDA)</h3>
            <p>Explora distribuciones, correlaciones y patrones en los datos.</p>
            <ul>
                <li>Estadísticas descriptivas</li>
                <li>Visualizaciones interactivas</li>
                <li>Análisis de correlaciones</li>
                <li>Detección de valores atípicos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="navigation-card">
            <h3>3️⃣ Evaluación del Modelo</h3>
            <p>Analiza el rendimiento del modelo entrenado.</p>
            <ul>
                <li>Métricas de precisión (R², RMSE, MAE)</li>
                <li>Análisis de residuos</li>
                <li>Gráficos de diagnóstico</li>
                <li>Validación cruzada</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="navigation-card">
            <h3>2️⃣ Entrenamiento del Modelo</h3>
            <p>Configura y entrena el modelo de regresión lineal.</p>
            <ul>
                <li>Selección de variables</li>
                <li>División de datos (train/test)</li>
                <li>Entrenamiento automático</li>
                <li>Visualización de coeficientes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="navigation-card">
            <h3>4️⃣ Predicciones</h3>
            <p>Realiza predicciones para nuevos estudiantes.</p>
            <ul>
                <li>Predicción individual</li>
                <li>Análisis de sensibilidad</li>
                <li>Interpretación de resultados</li>
                <li>Exportación de predicciones</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer simplificado
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
    <p>Sistema Predictivo de Rendimiento Académico • Desarrollado con Streamlit y Python</p>
</div>
""", unsafe_allow_html=True)