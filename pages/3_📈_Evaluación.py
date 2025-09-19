import streamlit as st
import pandas as pd
import numpy as np

# Configurar pandas para evitar problemas con PyArrow
pd.options.mode.string_storage = "python"
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                           explained_variance_score, max_error, mean_squared_log_error)
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Evaluación del Modelo",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado
st.markdown("""
<style>
    :root {
        --primary: #4A6FDC;
        --secondary: #6C63FF;
        --accent: #36D1DC;
        --light: #f8f9fa;
        --dark: #212529;
        --success: #28a745;
        --warning: #ffc107;
        --danger: #dc3545;
        --gray: #6c757d;
        --light-gray: #e9ecef;
    }

    .main {
        background-color: #fafbfe;
    }

    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 12px;
        color: white;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        height: 100%;
        border-top: 4px solid var(--primary);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }

    .metric-card h3 {
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
        color: var(--gray);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card h2 {
        font-size: 1.8rem;
        margin: 0;
        color: var(--dark);
        font-weight: 700;
    }

    .evaluation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin: 1rem 0;
        border-left: 4px solid var(--secondary);
    }

    .insight-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-left: 4px solid var(--warning);
    }

    .insight-box h4 {
        color: var(--warning);
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-left: 4px solid var(--success);
    }

    .success-box h4 {
        color: var(--success);
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .warning-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-left: 4px solid var(--danger);
    }

    .warning-box h4 {
        color: var(--danger);
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .tab-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .tip-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        margin: 0.8rem 0;
        border-left: 4px solid var(--accent);
    }

    .tip-card h4 {
        color: var(--accent);
        margin-top: 0;
        margin-bottom: 0.8rem;
        font-weight: 600;
        display: flex;
        align-items: center;
    }

    .tip-card h4 span {
        margin-right: 8px;
    }

    /* Mejora de pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: var(--light);
        border-radius: 8px 8px 0 0;
        gap: 8px;
        padding: 12px 20px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white;
    }

    /* Ajustes de espaciado */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Mejora de selectores */
    .stSelectbox div div {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header principal mejorado
st.markdown("""
<div class="main-header">
    <h1>📈 Evaluación Avanzada del Modelo Predictivo</h1>
    <p>Análisis completo del rendimiento del modelo con métricas avanzadas, diagnósticos y recomendaciones</p>
</div>
""", unsafe_allow_html=True)

try:
    # Cargar modelo y información
    loaded_model = joblib.load("modelo_regresion.pkl")

    # Verificar si el modelo cargado es una tupla o el modelo directamente
    if isinstance(loaded_model, tuple):
        # Si es una tupla, tomar el primer elemento que debería ser el modelo
        model = loaded_model[0]
        st.warning("⚠️ Modelo cargado desde formato legacy. Se recomienda reentrenar.")
    else:
        # Si es el modelo directamente
        model = loaded_model

    # Verificar que el modelo tiene el método predict
    if not hasattr(model, 'predict'):
        raise AttributeError("El objeto cargado no es un modelo válido")

    # Intentar cargar información del modelo
    try:
        model_info = joblib.load("modelo_info.pkl")
        model_name = model_info.get('model_name', 'Modelo Desconocido')
        features = model_info.get('features', [])
    except:
        model_name = "Modelo Cargado"
        features = []

    # Cargar y preparar datos
    df = pd.read_excel("dataset.xlsx")

    # Manejo robusto de la columna promedio según su tipo
    if df["promedio"].dtype == 'object':
        # Si es string, aplicar replace y convertir a float
        df["promedio"] = df["promedio"].str.replace(",", ".").astype(float)
    else:
        # Si ya es numérico, convertir directamente a float
        df["promedio"] = df["promedio"].astype(float)

    df_clean = df.drop(columns=["tipo_documento", "documento", "nombre_completo"], errors='ignore')

    # Si no tenemos información de características, usar todas las numéricas excepto promedio
    if not features:
        features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        features.remove('promedio')

    X = df_clean[features]
    y = df_clean["promedio"]

    # Dividir datos (usando la misma configuración que en entrenamiento)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Realizar predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Información del modelo
    st.markdown("""
    <div class="tab-container">
        <h2>🤖 Información del Modelo</h2>
        <p>Detalles del modelo entrenado y características utilizadas para las predicciones.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏷️ Tipo de Modelo</h3>
            <h2>{model_name}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Variables Predictoras</h3>
            <h2>{len(features)}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏋️ Muestras Entrenamiento</h3>
            <h2>{len(X_train)}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🧪 Muestras Prueba</h3>
            <h2>{len(X_test)}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Mostrar variables utilizadas
    with st.expander("📋 Ver variables utilizadas en el modelo"):
        st.write(", ".join(features))

    # Pestañas para organizar el contenido
    st.markdown("""
    <div class="tab-container">
        <h2>📊 Análisis de Evaluación del Modelo</h2>
        <p>Explora diferentes aspectos del rendimiento del modelo a través de las siguientes secciones:</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Métricas Principales", "🎯 Análisis de Predicciones", "📈 Diagnóstico de Residuos",
        "📉 Curvas de Aprendizaje", "🔍 Análisis Avanzado"
    ])

    # ==================== MÉTRICAS GENERALES ====================
    with tab1:
        st.markdown("""
        <div class="tab-container">
            <h2>📊 Métricas de Evaluación del Modelo</h2>
            <p>Métricas cuantitativas para evaluar el rendimiento del modelo en entrenamiento y prueba.</p>
        </div>
        """, unsafe_allow_html=True)


        # Calcular métricas para entrenamiento y prueba
        metrics_train = {
            'R²': r2_score(y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'MAE': mean_absolute_error(y_train, y_pred_train),
            'MSE': mean_squared_error(y_train, y_pred_train),
            'Varianza Explicada': explained_variance_score(y_train, y_pred_train),
            'Error Máximo': max_error(y_train, y_pred_train)
        }

        metrics_test = {
            'R²': r2_score(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'MSE': mean_squared_error(y_test, y_pred_test),
            'Varianza Explicada': explained_variance_score(y_test, y_pred_test),
            'Error Máximo': max_error(y_test, y_pred_test)
        }

        # Mostrar métricas principales
        st.subheader("🎯 Métricas Principales de Rendimiento")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>R² Entrenamiento</h3>
                <h2>{metrics_train['R²']:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>R² Prueba</h3>
                <h2>{metrics_test['R²']:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>RMSE Prueba</h3>
                <h2>{metrics_test['RMSE']:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>MAE Prueba</h3>
                <h2>{metrics_test['MAE']:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Tabla comparativa
        st.subheader("📋 Comparación Detallada: Entrenamiento vs Prueba")
        comparison_df = pd.DataFrame({
            'Métrica': list(metrics_train.keys()),
            'Entrenamiento': list(metrics_train.values()),
            'Prueba': list(metrics_test.values())
        }).round(4)

        comparison_df['Diferencia'] = (comparison_df['Entrenamiento'] - comparison_df['Prueba']).round(4)
        st.dataframe(comparison_df, use_container_width=True)

        # Gráfico de comparación
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Entrenamiento', x=comparison_df['Métrica'], y=comparison_df['Entrenamiento'],
                            marker_color=px.colors.qualitative.Set2[0]))
        fig.add_trace(go.Bar(name='Prueba', x=comparison_df['Métrica'], y=comparison_df['Prueba'],
                            marker_color=px.colors.qualitative.Set2[1]))
        fig.update_layout(title="Comparación de Métricas: Entrenamiento vs Prueba",
                         barmode='group', height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Interpretación de métricas
        overfitting_score = metrics_train['R²'] - metrics_test['R²']

        if overfitting_score > 0.1:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ Posible Sobreajuste Detectado</h4>
                <p>La diferencia entre R² de entrenamiento ({metrics_train['R²']:.4f}) y prueba ({metrics_test['R²']:.4f}) es significativa: {overfitting_score:.4f}.</p>
                <p><strong>Recomendaciones:</strong></p>
                <ul>
                    <li>Considera usar técnicas de regularización (Lasso, Ridge)</li>
                    <li>Reduce la complejidad del modelo</li>
                    <li>Aumenta el tamaño del conjunto de datos si es posible</li>
                    <li>Utiliza validación cruzada para ajustar hiperparámetros</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif overfitting_score < -0.05:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🤔 Comportamiento Inusual Detectado</h4>
                <p>El modelo tiene mejor rendimiento en prueba que en entrenamiento.</p>
                <p><strong>Posibles causas:</strong></p>
                <ul>
                    <li>Problemas en la división train-test</li>
                    <li>Características específicas del conjunto de prueba</li>
                    <li>Modelo demasiado simple (subajuste)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ Buen Balance del Modelo</h4>
                <p>El modelo muestra un buen balance entre entrenamiento y prueba.</p>
                <p>Diferencia en R²: {overfitting_score:.4f} (dentro del rango aceptable)</p>
                <p><strong>Interpretación:</strong> El modelo generaliza bien a datos no vistos.</p>
            </div>
            """, unsafe_allow_html=True)

    # ==================== ANÁLISIS DE PREDICCIONES ====================
    with tab2:
        st.markdown("""
        <div class="tab-container">
            <h2>🎯 Análisis Detallado de Predicciones</h2>
            <p>Visualización y análisis de las predicciones del modelo comparadas con los valores reales.</p>
        </div>
        """, unsafe_allow_html=True)


        # Gráficos de predicciones vs reales
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(x=y_train, y=y_pred_train,
                           title="Predicciones vs Reales (Entrenamiento)",
                           labels={'x': 'Valores Reales', 'y': 'Predicciones'},
                           trendline="ols",
                           color_discrete_sequence=[px.colors.qualitative.Set2[0]])
            fig.add_shape(type="line", x0=y_train.min(), y0=y_train.min(),
                         x1=y_train.max(), y1=y_train.max(),
                         line=dict(dash="dash", color="red"))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(x=y_test, y=y_pred_test,
                           title="Predicciones vs Reales (Prueba)",
                           labels={'x': 'Valores Reales', 'y': 'Predicciones'},
                           trendline="ols",
                           color_discrete_sequence=[px.colors.qualitative.Set2[1]])
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                         x1=y_test.max(), y1=y_test.max(),
                         line=dict(dash="dash", color="red"))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Análisis de errores por rangos
        st.subheader("📊 Análisis de Errores por Rangos de Valores")

        # Crear rangos de valores reales
        y_test_ranges = pd.cut(y_test, bins=5, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])
        error_by_range = pd.DataFrame({
            'Rango': y_test_ranges,
            'Error_Absoluto': np.abs(y_test - y_pred_test),
            'Error': y_test - y_pred_test
        })

        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(error_by_range, x='Rango', y='Error_Absoluto',
                        title="Error Absoluto por Rango de Valores",
                        color_discrete_sequence=[px.colors.qualitative.Set3[0]])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(error_by_range, x='Rango', y='Error',
                        title="Error (Sesgo) por Rango de Valores",
                        color_discrete_sequence=[px.colors.qualitative.Set3[1]])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Estadísticas de errores por rango
        st.subheader("📋 Estadísticas de Errores por Rango")
        error_stats = error_by_range.groupby('Rango').agg({
            'Error_Absoluto': ['mean', 'std', 'count'],
            'Error': ['mean', 'std']
        }).round(4)
        error_stats.columns = ['MAE', 'Std_Error_Abs', 'Cantidad', 'Sesgo_Promedio', 'Std_Error']
        st.dataframe(error_stats, use_container_width=True)

        # Análisis de patrones de error
        avg_bias = error_stats['Sesgo_Promedio'].mean()
        if abs(avg_bias) > 0.1:
            st.markdown(f"""
            <div class="insight-box">
                <h4>📊 Patrón de Sesgo Detectado</h4>
                <p>El modelo muestra un sesgo promedio de {avg_bias:.4f} puntos.</p>
                <p><strong>Interpretación:</strong> El modelo tiende a {'sobrestimar' if avg_bias < 0 else 'subestimar'} consistentemente los valores reales.</p>
                <p><strong>Recomendación:</strong> Considera ajustar el intercepto del modelo o aplicar transformaciones a la variable objetivo.</p>
            </div>
            """, unsafe_allow_html=True)

    # ==================== ANÁLISIS DE RESIDUOS ====================
    with tab3:
        st.markdown("""
        <div class="tab-container">
            <h2>📈 Análisis de Residuos del Modelo</h2>
            <p>Diagnóstico de los residuos para validar los supuestos del modelo de regresión lineal.</p>
        </div>
        """, unsafe_allow_html=True)


        # Calcular residuos
        residuals_train = y_train - y_pred_train
        residuals_test = y_test - y_pred_test

        # Gráficos de residuos
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(x=y_pred_train, y=residuals_train,
                           title="Residuos vs Predicciones (Entrenamiento)",
                           labels={'x': 'Predicciones', 'y': 'Residuos'},
                           color_discrete_sequence=[px.colors.qualitative.Set2[0]])
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(x=y_pred_test, y=residuals_test,
                           title="Residuos vs Predicciones (Prueba)",
                           labels={'x': 'Predicciones', 'y': 'Residuos'},
                           color_discrete_sequence=[px.colors.qualitative.Set2[1]])
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Distribución de residuos
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(x=residuals_test, nbins=20,
                             title="Distribución de Residuos (Prueba)",
                             marginal="box",
                             color_discrete_sequence=[px.colors.qualitative.Set3[2]])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Q-Q plot para normalidad de residuos
            fig, ax = plt.subplots(figsize=(8, 6))
            stats.probplot(residuals_test, dist="norm", plot=ax)
            ax.set_title("Q-Q Plot - Normalidad de Residuos")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # Test de normalidad de residuos
        shapiro_stat, shapiro_p = stats.shapiro(residuals_test)

        st.subheader("🧪 Tests Estadísticos de Residuos")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Test Shapiro-Wilk", f"{shapiro_stat:.4f}")
            st.metric("P-valor", f"{shapiro_p:.4f}")

        with col2:
            st.metric("Media de Residuos", f"{residuals_test.mean():.4f}")
            st.metric("Std de Residuos", f"{residuals_test.std():.4f}")

        with col3:
            st.metric("Asimetría", f"{stats.skew(residuals_test):.4f}")
            st.metric("Curtosis", f"{stats.kurtosis(residuals_test):.4f}")

        # Interpretación de residuos
        if shapiro_p > 0.05:
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ Residuos Normalmente Distribuidos</h4>
                <p>Los residuos siguen una distribución normal (p-valor = {shapiro_p:.4f} > 0.05)</p>
                <p><strong>Interpretación:</strong> El modelo cumple con el supuesto de normalidad de residuos.</p>
                <p><strong>Implicación:</strong> Los intervalos de confianza y pruebas de hipótesis son válidos.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ Residuos No Normales</h4>
                <p>Los residuos no siguen una distribución normal (p-valor = {shapiro_p:.4f} < 0.05)</p>
                <p><strong>Posibles causas:</strong></p>
                <ul>
                    <li>Presencia de outliers influyentes</li>
                    <li>Relaciones no lineales no capturadas</li>
                    <li>Heterocedasticidad (varianza no constante)</li>
                    <li>Falta de variables importantes</li>
                </ul>
                <p><strong>Recomendaciones:</strong></p>
                <ul>
                    <li>Aplicar transformaciones a la variable objetivo (log, sqrt)</li>
                    <li>Identificar y tratar outliers</li>
                    <li>Agregar términos no lineales al modelo</li>
                    <li>Considerar modelos robustos a no normalidad</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ==================== CURVAS DE APRENDIZAJE ====================
    with tab4:
        st.markdown("""
        <div class="tab-container">
            <h2>📉 Curvas de Aprendizaje y Validación</h2>
            <p>Análisis del comportamiento del modelo con diferentes tamaños de conjunto de entrenamiento.</p>
        </div>
        """, unsafe_allow_html=True)


        # Curva de aprendizaje
        st.subheader("📈 Curva de Aprendizaje")

        with st.spinner("Calculando curva de aprendizaje..."):
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='r2'
            )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        fig = go.Figure()

        # Entrenamiento
        fig.add_trace(go.Scatter(
            x=train_sizes, y=train_mean,
            mode='lines+markers',
            name='Entrenamiento',
            line=dict(color='blue'),
            error_y=dict(type='data', array=train_std, visible=True)
        ))

        # Validación
        fig.add_trace(go.Scatter(
            x=train_sizes, y=val_mean,
            mode='lines+markers',
            name='Validación',
            line=dict(color='red'),
            error_y=dict(type='data', array=val_std, visible=True)
        ))

        fig.update_layout(
            title="Curva de Aprendizaje",
            xaxis_title="Tamaño del Conjunto de Entrenamiento",
            yaxis_title="R² Score",
            hovermode='x',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Interpretación de la curva de aprendizaje
        final_gap = train_mean[-1] - val_mean[-1]

        if final_gap > 0.1:
            st.markdown(f"""
            <div class="warning-box">
                <h4>📊 Análisis de la Curva de Aprendizaje</h4>
                <p><strong>Brecha final:</strong> {final_gap:.4f} (significativa)</p>
                <p><strong>Diagnóstico:</strong> El modelo muestra signos de sobreajuste (alta varianza).</p>
                <p><strong>Recomendaciones:</strong></p>
                <ul>
                    <li>Reducir la complejidad del modelo</li>
                    <li>Aplicar técnicas de regularización (L1/L2)</li>
                    <li>Reducir el número de características</li>
                    <li>Aumentar el tamaño del conjunto de datos</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif final_gap < -0.05:
            st.markdown(f"""
            <div class="insight-box">
                <h4>📊 Análisis de la Curva de Aprendizaje</h4>
                <p><strong>Brecha final:</strong> {final_gap:.4f} (comportamiento inusual)</p>
                <p><strong>Diagnóstico:</strong> Mejor rendimiento en validación que en entrenamiento.</p>
                <p><strong>Posibles causas:</strong></p>
                <ul>
                    <li>Problemas en la división de datos</li>
                    <li>Características específicas del conjunto de validación</li>
                    <li>Modelo demasiado simple (subajuste)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <h4>📊 Análisis de la Curva de Aprendizaje</h4>
                <p><strong>Brecha final:</strong> {final_gap:.4f} (aceptable)</p>
                <p><strong>Diagnóstico:</strong> El modelo muestra un buen balance entre sesgo y varianza.</p>
                <p><strong>Interpretación:</strong> El modelo generaliza adecuadamente a datos no vistos.</p>
            </div>
            """, unsafe_allow_html=True)

    # ==================== DIAGNÓSTICOS AVANZADOS ====================
    with tab5:
        st.markdown("""
        <div class="tab-container">
            <h2>🔍 Diagnósticos Avanzados del Modelo</h2>
            <p>Análisis detallado de outliers, estabilidad del modelo y intervalos de confianza.</p>
        </div>
        """, unsafe_allow_html=True)

        # Análisis de outliers en predicciones
        st.subheader("🎯 Detección de Outliers en Predicciones")

        errors = np.abs(y_test - y_pred_test)
        Q1 = np.percentile(errors, 25)
        Q3 = np.percentile(errors, 75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR

        outliers_mask = errors > outlier_threshold
        outliers_df = pd.DataFrame({
            'Índice': y_test.index[outliers_mask],
            'Valor_Real': y_test[outliers_mask],
            'Predicción': y_pred_test[outliers_mask],
            'Error_Absoluto': errors[outliers_mask]
        }).sort_values('Error_Absoluto', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total de Outliers", len(outliers_df)),
            st.metric("Porcentaje de Outliers", f"{(len(outliers_df)/len(y_test)*100):.1f}%")
            st.metric("Umbral de Outlier", f"{outlier_threshold:.4f}")

        with col2:
            if len(outliers_df) > 0:
                fig = px.scatter(x=y_test, y=y_pred_test,
                               title="Outliers en Predicciones",
                               labels={'x': 'Valores Reales', 'y': 'Predicciones'},
                               color_discrete_sequence=[px.colors.qualitative.Set2[0]])

                # Marcar outliers
                fig.add_scatter(x=outliers_df['Valor_Real'], y=outliers_df['Predicción'],
                              mode='markers', marker=dict(color='red', size=10),
                              name='Outliers')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        if len(outliers_df) > 0:
            st.subheader("📋 Casos con Mayor Error de Predicción")
            st.dataframe(outliers_df.head(10), use_container_width=True)

            # Análisis de outliers
            st.markdown(f"""
            <div class="insight-box">
                <h4>💡 Análisis de Outliers</h4>
                <p>Se identificaron {len(outliers_df)} casos con error excepcionalmente alto.</p>
                <p><strong>Posibles causas:</strong></p>
                <ul>
                    <li>Errores de medición en los datos</li>
                    <li>Casos genuinamente atípicos en la población</li>
                    <li>Variables importantes no consideradas en el modelo</li>
                    <li>Relaciones no lineales no capturadas</li>
                </ul>
                <p><strong>Recomendaciones:</strong></p>
                <ul>
                    <li>Revisar la calidad de datos para estos casos</li>
                    <li>Considerar si son errores de medición o casos genuinos</li>
                    <li>Evaluar si se deben incluir en el modelo o tratarse por separado</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ Sin Outliers Detectados</h4>
                <p>No se detectaron casos con error excepcionalmente alto en las predicciones.</p>
                <p><strong>Interpretación:</strong> El modelo tiene un comportamiento consistente en todas las predicciones.</p>
            </div>
            """, unsafe_allow_html=True)

        # Análisis de estabilidad del modelo
        st.subheader("🔄 Análisis de Estabilidad con Bootstrap")

        # Bootstrap para intervalos de confianza
        n_bootstrap = 100
        bootstrap_r2 = []

        with st.spinner("Calculando intervalos de confianza con bootstrap..."):
            for _ in range(n_bootstrap):
                # Muestreo con reemplazo
                indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
                X_boot = X_test.iloc[indices]
                y_boot = y_test.iloc[indices]
                y_pred_boot = model.predict(X_boot)
                bootstrap_r2.append(r2_score(y_boot, y_pred_boot))

        r2_mean = np.mean(bootstrap_r2)
        r2_std = np.std(bootstrap_r2)
        r2_ci_lower = np.percentile(bootstrap_r2, 2.5)
        r2_ci_upper = np.percentile(bootstrap_r2, 97.5)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("R² Promedio Bootstrap", f"{r2_mean:.4f}")
        with col2:
            st.metric("Desv. Estándar Bootstrap", f"{r2_std:.4f}")
        with col3:
            st.metric("IC 95% Inferior", f"{r2_ci_lower:.4f}")
        with col4:
            st.metric("IC 95% Superior", f"{r2_ci_upper:.4f}")

        # Histograma de bootstrap
        fig = px.histogram(x=bootstrap_r2, nbins=20,
                          title="Distribución Bootstrap del R²",
                          color_discrete_sequence=[px.colors.qualitative.Set3[4]])
        fig.add_vline(x=r2_mean, line_dash="dash", line_color="red",
                     annotation_text=f"Media: {r2_mean:.4f}")
        fig.add_vline(x=r2_ci_lower, line_dash="dash", line_color="orange",
                     annotation_text=f"IC 2.5%: {r2_ci_lower:.4f}")
        fig.add_vline(x=r2_ci_upper, line_dash="dash", line_color="orange",
                     annotation_text=f"IC 97.5%: {r2_ci_upper:.4f}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Interpretación de estabilidad
        ci_width = r2_ci_upper - r2_ci_lower

        if ci_width > 0.2:
            st.markdown(f"""
            <div class="warning-box">
                <h4>📊 Estabilidad del Modelo</h4>
                <p><strong>Amplitud del IC 95%:</strong> {ci_width:.4f} (amplio)</p>
                <p><strong>Diagnóstico:</strong> El modelo muestra variabilidad significativa en diferentes muestras.</p>
                <p><strong>Recomendaciones:</strong></p>
                <ul>
                    <li>Considerar aumentar el tamaño de la muestra</li>
                    <li>Revisar la homogeneidad de los datos</li>
                    <li>Evaluar la posibilidad de sobreajuste</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <h4>📊 Estabilidad del Modelo</h4>
                <p><strong>Amplitud del IC 95%:</strong> {ci_width:.4f} (aceptable)</p>
                <p><strong>Diagnóstico:</strong> El modelo muestra buena estabilidad en diferentes muestras.</p>
                <p><strong>Interpretación:</strong> Las estimaciones del rendimiento son consistentes.</p>
            </div>
            """, unsafe_allow_html=True)

    # ==================== REPORTE COMPLETO ====================

except FileNotFoundError:
    st.error("""
    <div style='background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545;'>
        <h4>❌ Error: Dataset o modelo no encontrado</h4>
        <p>No se encontró el archivo 'dataset.xlsx' o 'modelo_regresion.pkl'. Asegúrate de:</p>
        <ul>
            <li>Tener el archivo 'dataset.xlsx' en el directorio</li>
            <li>Haber entrenado un modelo en la página 'Entrenamiento del Modelo'</li>
            <li>Verificar que los nombres de archivo sean correctos</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"""
    <div style='background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545;'>
        <h4>❌ Error inesperado</h4>
        <p>Ocurrió un error al evaluar el modelo: {str(e)}</p>
        <p>Verifica que el modelo haya sido entrenado correctamente y que los datos tengan el formato adecuado.</p>
    </div>
    """, unsafe_allow_html=True)





# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
    <p>Evaluación de Modelo Predictivo • Desarrollado con Streamlit y Scikit-learn • 2023</p>
</div>
""", unsafe_allow_html=True)