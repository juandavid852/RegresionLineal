import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Entrenamiento del Modelo", layout="wide")

# CSS personalizado
st.markdown("""
<style>
    .model-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 0.3rem;
        border-radius: 6px;
        text-align: center;
        margin: 0.15rem 0;
        min-height: 60px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        font-size: 0.7rem;
        margin-bottom: 0.2rem;
        color: #2e7d32;
        font-weight: 600;
        line-height: 1.2;
    }
    .metric-card h2 {
        font-size: 1.2rem;
        margin: 0;
        color: #1b5e20;
        font-weight: 700;
        line-height: 1.1;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Entrenamiento Avanzado del Modelo")
st.markdown("Entrena y compara diferentes modelos de machine learning para predecir el promedio académico.")

try:
    # Cargar y preparar datos
    df = pd.read_excel("dataset.xlsx")
    # Manejo robusto de la columna promedio
    if df["promedio"].dtype == 'object':
        # Si es string, reemplazar comas por puntos
        df["promedio"] = df["promedio"].str.replace(",", ".").astype(float)
    else:
        # Si ya es numérico, asegurar que sea float
        df["promedio"] = df["promedio"].astype(float)

    # Convertir tipos de datos para evitar errores de PyArrow
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'promedio':
            # Intentar convertir a numérico si es posible, excepto promedio que se maneja aparte
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                # Si no se puede convertir, mantener como string
                df[col] = df[col].astype(str)
        elif str(df[col].dtype).startswith('Int') or 'int64' in str(df[col].dtype):
            try:
                df[col] = df[col].fillna(0).astype('int64')
            except:
                df[col] = df[col].astype('object')

    df_clean = df.drop(columns=["tipo_documento", "documento", "nombre_completo"], errors='ignore')

    # Configuración con pestañas
    st.header("🔧 Configuración del Modelo")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 División de Datos",
        "📊 Variables",
        "⚙️ Preprocesamiento",
        "🤖 Modelos"
    ])

    # Variables por defecto
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('promedio')  # Remover variable objetivo

    with tab2:
        st.subheader("📊 Selección de Variables")
        st.markdown("**Variable Objetivo:** promedio (promedio académico del estudiante)")

        selected_features = st.multiselect(
            "Selecciona las variables predictoras:",
            numeric_cols,
            default=["horas_estudio_semana", "asistencia_clase"],
            help="Estas variables se usarán para predecir el promedio académico"
        )

        if len(selected_features) > 0:
            st.success(f"✅ {len(selected_features)} variables seleccionadas")
            with st.expander("Ver variables seleccionadas"):
                for i, var in enumerate(selected_features, 1):
                    st.write(f"{i}. **{var}**")
        else:
            st.error("❌ Debes seleccionar al menos una variable predictora.")

    with tab1:
        st.subheader("🔄 División de Datos")
        st.markdown("Configura cómo se dividirán los datos para entrenamiento y prueba.")

        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider(
                "Porcentaje para prueba (%)",
                10, 40, 20,
                help="Porcentaje de datos que se usarán para evaluar el modelo"
            ) / 100

        with col2:
            random_state = st.number_input(
                "Semilla aleatoria",
                1, 100, 42,
                help="Garantiza resultados reproducibles"
            )

        # Mostrar información de la división
        total_records = len(df_clean)
        train_size = int(total_records * (1 - test_size))
        test_size_records = total_records - train_size

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Datos de Entrenamiento", f"{train_size:,}")
        with col2:
            st.metric("Datos de Prueba", f"{test_size_records:,}")

    with tab3:
        st.subheader("⚙️ Preprocesamiento de Datos")
        st.markdown("Configura las transformaciones que se aplicarán a los datos antes del entrenamiento.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔢 Escalado de Variables**")
            scale_features = st.checkbox(
                "Escalar variables",
                value=True,
                help="Normaliza las variables para que tengan media 0 y desviación estándar 1"
            )

            if scale_features:
                st.success("✅ Se aplicará StandardScaler")
            else:
                st.info("ℹ️ No se escalará las variables")

        with col2:
            st.markdown("**📈 Características Polinómicas**")
            polynomial_features = st.checkbox(
                "Generar características polinómicas",
                value=False,
                help="Crea nuevas variables combinando las existentes (ej: x₁², x₁×x₂)"
            )

            if polynomial_features:
                poly_degree = st.slider(
                    "Grado del polinomio",
                    2, 4, 2,
                    help="Grado máximo de las características polinómicas"
                )
                st.success(f"✅ Se generarán características de grado {poly_degree}")
            else:
                poly_degree = 2  # Valor por defecto
                st.info("ℹ️ Solo se usarán las variables originales")

        # Información adicional
        with st.expander("ℹ️ Información sobre Preprocesamiento"):
            st.markdown("""
            **Escalado de Variables:**
            - Recomendado cuando las variables tienen diferentes escalas
            - Mejora el rendimiento de algoritmos como SVM y redes neuronales
            - No afecta a modelos basados en árboles (Random Forest, Gradient Boosting)

            **Características Polinómicas:**
            - Permite capturar relaciones no lineales
            - Aumenta la complejidad del modelo
            - Puede causar sobreajuste con grados altos
            """)

    with tab4:
        st.subheader("🤖 Selección de Modelos")
        st.markdown("Elige los algoritmos de machine learning que quieres entrenar y comparar.")

        # Definir modelos disponibles
        model_options = {
            "Regresión Lineal": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Elastic Net": ElasticNet(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVM": SVR()
        }

        selected_models = st.multiselect(
            "Selecciona los modelos a entrenar:",
            list(model_options.keys()),
            default=["Regresión Lineal", "Ridge", "Random Forest"],
            help="Puedes seleccionar múltiples modelos para compararlos"
        )

        if len(selected_models) > 0:
            st.success(f"✅ {len(selected_models)} modelos seleccionados")

            # Mostrar información de los modelos seleccionados
            with st.expander("Información de los modelos seleccionados"):
                model_info = {
                    "Regresión Lineal": "Modelo básico que encuentra la mejor línea recta",
                    "Ridge": "Regresión lineal con regularización L2",
                    "Lasso": "Regresión lineal con regularización L1",
                    "Elastic Net": "Combina regularización L1 y L2",
                    "Random Forest": "Ensemble de árboles de decisión",
                    "Gradient Boosting": "Ensemble secuencial de modelos débiles",
                    "SVM": "Máquinas de vectores de soporte para regresión"
                }

                for model in selected_models:
                    st.write(f"**{model}:** {model_info.get(model, 'Modelo de machine learning')}")
        else:
            st.error("❌ Debes seleccionar al menos un modelo.")


    # Validaciones antes de continuar
    if len(selected_features) == 0:
        st.error("❌ Debes seleccionar al menos una variable predictora en la pestaña 'Variables'.")
        st.stop()

    if len(selected_models) == 0:
        st.error("❌ Debes seleccionar al menos un modelo en la pestaña 'Modelos'.")
        st.stop()

    # Preparar datos
    X = df_clean[selected_features]
    y = df_clean["promedio"]

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


    # Mostrar variables seleccionadas
    st.subheader("🎯 Variables Seleccionadas")
    st.write("**Variable Objetivo:** promedio")
    st.write("**Variables Predictoras:**", ", ".join(selected_features))

    # Botón para entrenar modelos
    if st.button("🚀 Entrenar Modelos", type="primary"):

        # Crear pipeline de preprocesamiento
        preprocessing_steps = []

        if polynomial_features:
            preprocessing_steps.append(('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)))

        if scale_features:
            preprocessing_steps.append(('scaler', StandardScaler()))

        # Entrenar modelos
        results = {}
        trained_models = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, model_name in enumerate(selected_models):
            status_text.text(f"Entrenando {model_name}...")

            # Crear pipeline
            pipeline_steps = preprocessing_steps + [('model', model_options[model_name])]
            pipeline = Pipeline(pipeline_steps)

            # Entrenar modelo
            pipeline.fit(X_train, y_train)

            # Predicciones
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)

            # Métricas
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)

            # Validación cruzada
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')

            results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }

            trained_models[model_name] = pipeline

            progress_bar.progress((i + 1) / len(selected_models))

        status_text.text("✅ Entrenamiento completado!")

        # Mostrar resultados
        st.header("📈 Resultados del Entrenamiento")

        # Tabla de comparación
        st.subheader("📊 Comparación de Modelos")
        comparison_df = pd.DataFrame({
            'Modelo': list(results.keys()),
            'R² Entrenamiento': [results[m]['train_r2'] for m in results.keys()],
            'R² Prueba': [results[m]['test_r2'] for m in results.keys()],
            'RMSE Prueba': [results[m]['test_rmse'] for m in results.keys()],
            'MAE Prueba': [results[m]['test_mae'] for m in results.keys()],
            'CV R² (Media)': [results[m]['cv_mean'] for m in results.keys()],
            'CV R² (Std)': [results[m]['cv_std'] for m in results.keys()]
        }).round(4)

        st.dataframe(comparison_df, use_container_width=True)

        # Gráfico de comparación
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(comparison_df, x='Modelo', y='R² Prueba',
                        title="R² en Datos de Prueba por Modelo",
                        color='R² Prueba', color_continuous_scale="Viridis")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(comparison_df, x='Modelo', y='RMSE Prueba',
                        title="RMSE en Datos de Prueba por Modelo",
                        color='RMSE Prueba', color_continuous_scale="Reds_r")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Seleccionar mejor modelo
        best_model_name = comparison_df.loc[comparison_df['R² Prueba'].idxmax(), 'Modelo']
        best_model = trained_models[best_model_name]

        st.markdown(f"""
        <div class="model-card">
            <h3>🏆 Mejor Modelo: {best_model_name}</h3>
            <p><strong>R² en Prueba:</strong> {results[best_model_name]['test_r2']:.4f}</p>
            <p><strong>RMSE en Prueba:</strong> {results[best_model_name]['test_rmse']:.4f}</p>
            <p><strong>MAE en Prueba:</strong> {results[best_model_name]['test_mae']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Guardar mejor modelo
        joblib.dump(best_model, "modelo_regresion.pkl")

        # Guardar información del modelo
        model_info = {
            'model_name': best_model_name,
            'features': selected_features,
            'test_r2': results[best_model_name]['test_r2'],
            'test_rmse': results[best_model_name]['test_rmse'],
            'test_mae': results[best_model_name]['test_mae'],
            'preprocessing': {
                'scale_features': scale_features,
                'polynomial_features': polynomial_features,
                'poly_degree': poly_degree if polynomial_features else None
            }
        }
        joblib.dump(model_info, "modelo_info.pkl")

        st.success(f"✅ Mejor modelo ({best_model_name}) guardado exitosamente!")

        # Análisis detallado del mejor modelo
        st.subheader(f"🔍 Análisis Detallado - {best_model_name}")

        # Gráfico de predicciones vs reales
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(x=y_train, y=results[best_model_name]['y_pred_train'],
                           title="Predicciones vs Reales (Entrenamiento)",
                           labels={'x': 'Valores Reales', 'y': 'Predicciones'})
            fig.add_shape(type="line", x0=y_train.min(), y0=y_train.min(),
                         x1=y_train.max(), y1=y_train.max(),
                         line=dict(dash="dash", color="red"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(x=y_test, y=results[best_model_name]['y_pred_test'],
                           title="Predicciones vs Reales (Prueba)",
                           labels={'x': 'Valores Reales', 'y': 'Predicciones'})
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                         x1=y_test.max(), y1=y_test.max(),
                         line=dict(dash="dash", color="red"))
            st.plotly_chart(fig, use_container_width=True)

        # Residuos
        residuals_train = y_train - results[best_model_name]['y_pred_train']
        residuals_test = y_test - results[best_model_name]['y_pred_test']

        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(x=results[best_model_name]['y_pred_train'], y=residuals_train,
                           title="Residuos vs Predicciones (Entrenamiento)",
                           labels={'x': 'Predicciones', 'y': 'Residuos'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(x=results[best_model_name]['y_pred_test'], y=residuals_test,
                           title="Residuos vs Predicciones (Prueba)",
                           labels={'x': 'Predicciones', 'y': 'Residuos'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        # Importancia de características (si es aplicable)
        if hasattr(best_model.named_steps['model'], 'feature_importances_'):
            st.subheader("📊 Importancia de las Variables")

            # Obtener nombres de características después del preprocesamiento
            feature_names = selected_features
            if polynomial_features:
                poly_transformer = best_model.named_steps['poly']
                feature_names = poly_transformer.get_feature_names_out(selected_features)

            importances = best_model.named_steps['model'].feature_importances_

            importance_df = pd.DataFrame({
                'Variable': feature_names,
                'Importancia': importances
            }).sort_values('Importancia', ascending=False)

            fig = px.bar(importance_df.head(10), x='Importancia', y='Variable',
                        orientation='h', title="Top 10 Variables Más Importantes")
            st.plotly_chart(fig, use_container_width=True)

        elif hasattr(best_model.named_steps['model'], 'coef_'):
            st.subheader("📊 Coeficientes del Modelo")

            # Obtener nombres de características después del preprocesamiento
            feature_names = selected_features
            if polynomial_features:
                poly_transformer = best_model.named_steps['poly']
                feature_names = poly_transformer.get_feature_names_out(selected_features)

            coef = best_model.named_steps['model'].coef_

            coef_df = pd.DataFrame({
                'Variable': feature_names,
                'Coeficiente': coef
            }).sort_values('Coeficiente', key=abs, ascending=False)

            fig = px.bar(coef_df.head(10), x='Coeficiente', y='Variable',
                        orientation='h', title="Top 10 Coeficientes Más Importantes",
                        color='Coeficiente', color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Configura los parámetros en el sidebar y haz clic en 'Entrenar Modelos' para comenzar.")

except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'dataset.xlsx'. Asegúrate de que esté en el directorio correcto.")
except Exception as e:
    st.error(f"❌ Error al procesar los datos: {str(e)}")
