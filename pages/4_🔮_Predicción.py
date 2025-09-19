import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predicción Interactiva", layout="wide")

# CSS personalizado
st.markdown("""
<style>
    .prediction-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        border-left: 6px solid #4caf50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .input-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .batch-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ff9800;
    }
    .analysis-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
    }
    .metric-small {
        background: white;
        padding: 0.8rem;
        border-radius: 6px;
        text-align: center;
        margin: 0.3rem;
        border: 1px solid #e0e0e0;
    }
    .tab-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Predicción Interactiva Avanzada")
st.markdown("Realiza predicciones individuales o en lote con análisis detallado y visualizaciones.")

try:
    # Cargar modelo y información
    model = joblib.load("modelo_regresion.pkl")

    # Intentar cargar información del modelo
    try:
        model_info = joblib.load("modelo_info.pkl")
        model_name = model_info.get('model_name', 'Modelo Desconocido')
        features = model_info.get('features', [])
    except:
        model_name = "Modelo Cargado"
        features = []

    # Cargar y preparar datos para referencia
    df = pd.read_excel("dataset.xlsx", sheet_name="Dataset")

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

    # Calcular estadísticas de referencia
    feature_stats = X.describe()

    # Sistema de pestañas para organizar toda la funcionalidad
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Ingreso de Datos",
        "📊 Resultados y Análisis",
        "📈 Gráfica de Regresión",
        "🔍 Análisis de Sensibilidad"
    ])

    # ==================== INGRESO DE DATOS ====================
    with tab1:
        st.header("📝 Ingreso de Datos para Predicción")

        st.markdown("""
        <div class="analysis-card">
            <h4>💡 Instrucciones</h4>
            <p>Ingresa los valores para cada variable predictora utilizando los controles deslizantes.</p>
            <p>Los valores se inicializan con el promedio histórico de cada variable.</p>
        </div>
        """, unsafe_allow_html=True)

        # Crear inputs para cada variable con información estadística
        input_values = {}

        for i, feature in enumerate(features):
            # Obtener estadísticas de la variable
            min_val = float(feature_stats.loc['min', feature])
            max_val = float(feature_stats.loc['max', feature])
            mean_val = float(feature_stats.loc['mean', feature])

            col1, col2 = st.columns([3, 1])

            with col1:
                input_values[feature] = st.slider(
                    f"**{feature}**",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    help=f"Rango: {min_val:.2f} - {max_val:.2f}, Promedio: {mean_val:.2f}"
                )

            with col2:
                st.markdown(f"""
                <div class="metric-small">
                    <small>Min: {min_val:.1f}</small><br>
                    <small>Max: {max_val:.1f}</small><br>
                    <small>Prom: {mean_val:.1f}</small>
                </div>
                """, unsafe_allow_html=True)

        # Realizar predicción
        input_array = np.array([list(input_values.values())])
        prediction = model.predict(input_array)[0]

        # Guardar la predicción en session state para usar en otras pestañas
        st.session_state.prediction = prediction
        st.session_state.input_values = input_values

        # Mostrar predicción
        st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 Predicción Calculada</h2>
                <h1 style="color: #4caf50; font-size: 3em;">{prediction:.2f}</h1>
                <p style="font-size: 1.2em;">Promedio Estimado</p>
            </div>
            """, unsafe_allow_html=True)

        st.success("✅ Predicción calculada correctamente. Navega a las otras pestañas para ver análisis detallados.")

    # ==================== RESULTADOS Y ANÁLISIS ====================
    with tab2:
        st.header("📊 Resultados y Análisis de la Predicción")

        # Verificar si hay una predicción guardada
        if 'prediction' not in st.session_state:
            st.warning("⚠️ Por favor, realiza primero una predicción en la pestaña 'Ingreso de Datos'.")
        else:
            prediction = st.session_state.prediction
            input_values = st.session_state.input_values

            col1, col2 = st.columns(2)

            with col1:
                # Mostrar predicción principal
                st.markdown(f"""
                    <div class="prediction-card">
                        <h2>🎯 Predicción</h2>
                        <h1 style="color: #4caf50; font-size: 3em;">{prediction:.2f}</h1>
                        <p style="font-size: 1.2em;">Promedio Estimado</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Calcular percentil de la predicción
                percentile = (y <= prediction).mean() * 100

                st.markdown(f"""
                    <div class="analysis-card">
                        <h4>📊 Análisis de la Predicción</h4>
                        <p><strong>Percentil:</strong> {percentile:.1f}%</p>
                        <p>Esta predicción está por encima del {percentile:.1f}% de los datos históricos.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                # Clasificación de la predicción
                if prediction >= 4.5:
                    categoria = "🏆 Excelente"
                    color = "#4caf50"
                elif prediction >= 4.0:
                    categoria = "✅ Muy Bueno"
                    color = "#8bc34a"
                elif prediction >= 3.5:
                    categoria = "👍 Bueno"
                    color = "#ffc107"
                elif prediction >= 3.0:
                    categoria = "⚠️ Regular"
                    color = "#ff9800"
                else:
                    categoria = "❌ Bajo"
                    color = "#f44336"

                st.markdown(f"""
                    <div style="background: {color}20; padding: 1.5rem; border-radius: 8px; border-left: 4px solid {color}; margin-bottom: 20px;">
                        <h3 style="color: {color}; margin-top: 0;">{categoria}</h3>
                        <p>Clasificación basada en el promedio predicho</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Mostrar valores ingresados
                st.markdown("**📋 Valores ingresados:**")
                for feature, value in input_values.items():
                    st.write(f"- {feature}: {value:.2f}")

            # Gráfico de comparación con distribución histórica
            st.subheader("📈 Comparación con Datos Históricos")

            fig = go.Figure()

            # Histograma de datos históricos
            fig.add_trace(go.Histogram(
                x=y,
                name="Datos Históricos",
                opacity=0.7,
                nbinsx=20,
                marker_color='lightblue'
            ))

            # Línea vertical para la predicción
            fig.add_vline(
                x=prediction,
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text=f"Predicción: {prediction:.2f}",
                annotation_position="top"
            )

            fig.update_layout(
                title="Distribución de Promedios Históricos vs Predicción",
                xaxis_title="Promedio",
                yaxis_title="Frecuencia",
                showlegend=True,
                bargap=0.1
            )

            st.plotly_chart(fig, use_container_width=True)

    # ==================== GRÁFICA DE REGRESIÓN ====================
    with tab3:
        st.header("📈 Gráfica de Regresión Lineal")

        # Verificar si hay una predicción guardada
        if 'prediction' not in st.session_state:
            st.warning("⚠️ Por favor, realiza primero una predicción en la pestaña 'Ingreso de Datos'.")
        else:
            st.markdown("""
            <div class="analysis-card">
                <h4>📖 ¿Qué es una gráfica de regresión?</h4>
                <p>Una gráfica de regresión muestra la relación entre una variable independiente (predictora) y la variable dependiente (objetivo).</p>
                <p>Esta visualización te ayuda a entender cómo cada variable afecta el resultado predicho.</p>
            </div>
            """, unsafe_allow_html=True)

            # Seleccionar variable para la gráfica de regresión
            selected_feature = st.selectbox("Selecciona la variable para visualizar:", features)

            # Crear gráfica de dispersión con línea de regresión
            fig = px.scatter(
                df_clean,
                x=selected_feature,
                y='promedio',
                title=f"Relación entre {selected_feature} y el Promedio",
                trendline="ols",  # Línea de regresión lineal
                trendline_color_override="red"
            )

            # Añadir punto de la predicción actual
            prediction_val = st.session_state.prediction
            input_val = st.session_state.input_values[selected_feature]

            fig.add_trace(go.Scatter(
                x=[input_val],
                y=[prediction_val],
                mode='markers',
                marker=dict(color='green', size=12, symbol='star'),
                name='Predicción Actual'
            ))

            fig.update_layout(
                xaxis_title=selected_feature,
                yaxis_title="Promedio",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Mostrar ecuación de regresión
            st.markdown("**📐 Ecuación de regresión:**")

            # Manejar tanto modelos simples como Pipeline
            try:
                if hasattr(model, 'named_steps'):
                    # Es un Pipeline
                    actual_model = model.named_steps['model']
                    if hasattr(actual_model, 'intercept_') and hasattr(actual_model, 'coef_'):
                        intercept = actual_model.intercept_
                        coef = actual_model.coef_[features.index(selected_feature)]
                        st.latex(f"promedio = {intercept:.2f} + {coef:.2f} \\times {selected_feature}")
                    else:
                        st.info("📝 La ecuación de regresión no está disponible para este tipo de modelo.")
                else:
                    # Es un modelo simple
                    if hasattr(model, 'intercept_') and hasattr(model, 'coef_'):
                        intercept = model.intercept_
                        coef = model.coef_[features.index(selected_feature)]
                        st.latex(f"promedio = {intercept:.2f} + {coef:.2f} \\times {selected_feature}")
                    else:
                        st.info("📝 La ecuación de regresión no está disponible para este tipo de modelo.")
            except Exception as e:
                st.info("📝 La ecuación de regresión no está disponible para este modelo.")

    # ==================== ANÁLISIS DE SENSIBILIDAD ====================
    with tab4:
        st.header("🔍 Análisis de Sensibilidad")

        # Verificar si hay una predicción guardada
        if 'prediction' not in st.session_state:
            st.warning("⚠️ Por favor, realiza primero una predicción en la pestaña 'Ingreso de Datos'.")
        else:
            st.markdown("""
            <div class="analysis-card">
                <h4>🎯 ¿Qué es el Análisis de Sensibilidad?</h4>
                <p>Analiza cómo cambia la predicción cuando modificas cada variable individualmente.</p>
                <p>Esto te ayuda a entender qué variables tienen mayor impacto en el resultado.</p>
            </div>
            """, unsafe_allow_html=True)

            # Valores base para el análisis
            st.subheader("📊 Configuración Base")

            base_values = st.session_state.input_values.copy()
            cols = st.columns(min(3, len(features)))

            for i, feature in enumerate(features):
                with cols[i % 3]:
                    mean_val = float(feature_stats.loc['mean', feature])
                    base_values[feature] = st.number_input(
                        f"**{feature}**",
                        value=base_values[feature],
                        key=f"base_{feature}"
                    )

            # Predicción base
            base_array = np.array([list(base_values.values())])
            base_prediction = model.predict(base_array)[0]

            st.info(f"📈 Predicción con valores actuales: **{base_prediction:.2f}**")

            # Análisis de sensibilidad
            st.subheader("📈 Análisis de Sensibilidad por Variable")

            selected_feature = st.selectbox("Selecciona la variable a analizar:", features, key="sensitivity_feature")

            # Rango de valores para el análisis
            min_val = float(feature_stats.loc['min', selected_feature])
            max_val = float(feature_stats.loc['max', selected_feature])

            # Generar rango de valores
            feature_range = np.linspace(min_val, max_val, 50)
            sensitivity_predictions = []

            for value in feature_range:
                temp_values = base_values.copy()
                temp_values[selected_feature] = value
                temp_array = np.array([list(temp_values.values())])
                pred = model.predict(temp_array)[0]
                sensitivity_predictions.append(pred)

            # Gráfico de sensibilidad
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=feature_range,
                y=sensitivity_predictions,
                mode='lines',
                name=f'Predicción vs {selected_feature}',
                line=dict(width=3, color='blue')
            ))

            # Marcar el valor base
            fig.add_scatter(
                x=[base_values[selected_feature]],
                y=[base_prediction],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Valor Base'
            )

            fig.update_layout(
                title=f"Sensibilidad de la Predicción a {selected_feature}",
                xaxis_title=selected_feature,
                yaxis_title="Predicción",
                hovermode='x'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Análisis de impacto
            impact = max(sensitivity_predictions) - min(sensitivity_predictions)
            st.metric("Impacto máximo de esta variable", f"{impact:.2f} puntos")

            # Análisis de todas las variables
            st.subheader("🎯 Ranking de Importancia de Variables")

            with st.spinner("Calculando importancia de todas las variables..."):
                variable_impacts = {}

                for feature in features:
                    min_val = float(feature_stats.loc['min', feature])
                    max_val = float(feature_stats.loc['max', feature])
                    feature_range = np.linspace(min_val, max_val, 20)

                    predictions = []
                    for value in feature_range:
                        temp_values = base_values.copy()
                        temp_values[feature] = value
                        temp_array = np.array([list(temp_values.values())])
                        pred = model.predict(temp_array)[0]
                        predictions.append(pred)

                    variable_impacts[feature] = max(predictions) - min(predictions)

            # Crear DataFrame de importancia
            importance_df = pd.DataFrame({
                'Variable': list(variable_impacts.keys()),
                'Impacto': list(variable_impacts.values())
            }).sort_values('Impacto', ascending=False)

            # Gráfico de importancia
            fig = px.bar(
                importance_df,
                x='Impacto',
                y='Variable',
                orientation='h',
                title="Ranking de Importancia de Variables",
                color='Impacto',
                color_continuous_scale='viridis'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Tabla de importancia
            st.dataframe(importance_df, use_container_width=True)

except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'dataset.xlsx' o el modelo entrenado.")
    st.info("💡 Asegúrate de:")
    st.markdown("• Tener el archivo 'dataset.xlsx' en el directorio")
    st.markdown("• Haber entrenado un modelo en la página 'Modelo'")

except Exception as e:
    st.error(f"❌ Error al realizar predicciones: {str(e)}")
    st.info("💡 Verifica que el modelo haya sido entrenado correctamente.")