import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="EDA - Análisis Exploratorio",
    layout="wide",
    page_icon="📊",
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

    .insight-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-left: 4px solid var(--primary);
    }

    .insight-box h4 {
        color: var(--primary);
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
        border-left: 4px solid var(--success);
    }

    .tip-card h4 {
        color: var(--success);
        margin-top: 0;
        margin-bottom: 0.8rem;
        font-weight: 600;
        display: flex;
        align-items: center;
    }

    .tip-card h4 span {
        margin-right: 8px;
    }

    .warning-card {
        background: #fff3cd;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        margin: 0.8rem 0;
        border-left: 4px solid var(--warning);
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
    <h1>📊 Análisis Exploratorio de Datos (EDA)</h1>
    <p>Explora en profundidad el dataset de estudiantes y descubre patrones clave que influyen en el rendimiento académico</p>
</div>
""", unsafe_allow_html=True)

try:
    # Cargar y limpiar dataset
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

    df_clean = df.drop(columns=["tipo_documento", "documento"], errors='ignore')


    # Opciones de análisis en pestañas
    tab_options = ["📋 Resumen General", "📊 Distribuciones", "🔗 Correlaciones",
                 "📈 Análisis por Categorías", "🎯 Análisis del Promedio", "🔍 Valores Atípicos"]

    tabs = st.tabs(tab_options)

    # ==================== RESUMEN GENERAL ====================
    with tabs[0]:
        st.markdown("""
        <div class="tab-container">
            <h2>📋 Resumen General del Dataset</h2>
            <p>Vista completa de la estructura y características principales del dataset de estudiantes.</p>
        </div>
        """, unsafe_allow_html=True)


        # Vista previa del dataset
        st.subheader("👀 Vista Previa del Dataset")
        st.dataframe(df_clean.head(10), use_container_width=True)

        # Información de las variables
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Información de Variables")
            info_df = pd.DataFrame({
                'Variable': df_clean.columns,
                'Tipo': df_clean.dtypes,
                'Valores Únicos': [df_clean[col].nunique() for col in df_clean.columns],
                'Valores Faltantes': df_clean.isnull().sum(),
                '% Faltantes': [(df_clean[col].isnull().sum() / len(df_clean)) * 100 for col in df_clean.columns]
            })
            info_df['% Faltantes'] = info_df['% Faltantes'].round(2)
            st.dataframe(info_df, use_container_width=True)

        with col2:
            st.subheader("📈 Estadísticas Descriptivas")
            st.dataframe(df_clean.describe().round(2), use_container_width=True)

            # Detección de problemas de calidad
            st.subheader("🔍 Calidad de Datos")
            quality_issues = []

            # Verificar variables constantes
            constant_vars = [col for col in df_clean.columns if df_clean[col].nunique() == 1]
            if constant_vars:
                quality_issues.append(f"Variables constantes: {', '.join(constant_vars)}")

            # Verificar alto porcentaje de missing values
            high_missing = [col for col in df_clean.columns if (df_clean[col].isnull().sum() / len(df_clean)) > 0.3]
            if high_missing:
                quality_issues.append(f"Alto % de valores faltantes: {', '.join(high_missing)}")

            if quality_issues:
                st.warning("⚠️ **Problemas de calidad detectados:**")
                for issue in quality_issues:
                    st.write(f"- {issue}")
            else:
                st.success("✅ No se detectaron problemas graves de calidad de datos")

    # ==================== DISTRIBUCIONES ====================
    with tabs[1]:
        st.markdown("""
        <div class="tab-container">
            <h2>📊 Análisis de Distribuciones</h2>
            <p>Explora la distribución de variables numéricas, normalidad y características estadísticas.</p>
        </div>
        """, unsafe_allow_html=True)

        # Selección de variable
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        selected_var = st.selectbox("Selecciona una variable numérica:", numeric_cols)


        col1, col2 = st.columns(2)

        with col1:
            # Histograma con Plotly
            fig = px.histogram(df_clean, x=selected_var, nbins=20,
                             title=f"Distribución de {selected_var}",
                             marginal="box",
                             color_discrete_sequence=[px.colors.qualitative.Set2[0]])
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Q-Q Plot para normalidad
            fig, ax = plt.subplots(figsize=(8, 6))
            stats.probplot(df_clean[selected_var].dropna(), dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot - {selected_var}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # Estadísticas de la variable
        st.subheader(f"📊 Estadísticas de {selected_var}")
        col1, col2, col3, col4 = st.columns(4)

        var_data = df_clean[selected_var].dropna()

        with col1:
            st.metric("Media", f"{var_data.mean():.2f}")
            st.metric("Mediana", f"{var_data.median():.2f}")

        with col2:
            st.metric("Desv. Estándar", f"{var_data.std():.2f}")
            st.metric("Varianza", f"{var_data.var():.2f}")

        with col3:
            st.metric("Mínimo", f"{var_data.min():.2f}")
            st.metric("Máximo", f"{var_data.max():.2f}")

        with col4:
            st.metric("Asimetría", f"{stats.skew(var_data):.2f}")
            st.metric("Curtosis", f"{stats.kurtosis(var_data):.2f}")

        # Test de normalidad
        shapiro_stat, shapiro_p = stats.shapiro(var_data)

        # Interpretación de normalidad
        normality_text = "Los datos siguen una distribución normal (p > 0.05)" if shapiro_p > 0.05 else "Los datos NO siguen una distribución normal (p ≤ 0.05)"
        normality_color = "success" if shapiro_p > 0.05 else "warning"

        st.markdown(f"""
        <div class="insight-box">
            <h4>🧪 Test de Normalidad (Shapiro-Wilk)</h4>
            <p><strong>Estadístico:</strong> {shapiro_stat:.4f}</p>
            <p><strong>P-valor:</strong> {shapiro_p:.4f}</p>
            <p><strong>Interpretación:</strong> {normality_text}</p>
            <p><strong>Recomendación:</strong> {"No se requieren transformaciones" if shapiro_p > 0.05 else "Considerar transformaciones (log, sqrt) para normalizar"}</p>
        </div>
        """, unsafe_allow_html=True)

    # ==================== CORRELACIONES ====================
    with tabs[2]:
        st.markdown("""
        <div class="tab-container">
            <h2>🔗 Análisis de Correlaciones</h2>
            <p>Identifica relaciones entre variables y su influencia en el promedio académico.</p>
        </div>
        """, unsafe_allow_html=True)


        # Matriz de correlación interactiva
        numeric_df = df_clean.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        # Heatmap con Plotly
        fig = px.imshow(corr_matrix,
                       text_auto=True,
                       aspect="auto",
                       title="Matriz de Correlación",
                       color_continuous_scale="RdBu_r",
                       zmin=-1, zmax=1)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Correlaciones más fuertes con el promedio
        st.subheader("🎯 Correlaciones con el Promedio Académico")

        if 'promedio' in corr_matrix.columns:
            # Asegurar que sea una Series de pandas para usar .abs()
            promedio_series = pd.Series(corr_matrix['promedio'], index=corr_matrix.index)
            promedio_corr = promedio_series.abs().sort_values(ascending=False)[1:]  # Excluir autocorrelación
        else:
            st.error("La columna 'promedio' no se encuentra en los datos numéricos.")
            promedio_corr = pd.Series(dtype=float)

        col1, col2 = st.columns(2)

        if 'promedio' in corr_matrix.columns and len(promedio_corr) > 0:
            with col1:
                # Top correlaciones positivas
                positive_series = pd.Series(corr_matrix['promedio'], index=corr_matrix.index)
                positive_corr = positive_series[positive_series > 0].sort_values(ascending=False)[1:]
                if len(positive_corr) > 0:
                    fig = px.bar(x=positive_corr.values, y=positive_corr.index,
                               orientation='h', title="Correlaciones Positivas con el Promedio",
                               color=positive_corr.values, color_continuous_scale="Greens",
                               labels={'x': 'Coeficiente de Correlación', 'y': 'Variable'})
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No se encontraron correlaciones positivas significativas")

            with col2:
                # Top correlaciones negativas
                negative_series = pd.Series(corr_matrix['promedio'], index=corr_matrix.index)
                negative_corr = negative_series[negative_series < 0].sort_values()
                if len(negative_corr) > 0:
                    # Usar np.abs() en lugar de .abs() para mayor compatibilidad
                    fig = px.bar(x=np.abs(negative_corr.values), y=negative_corr.index,
                               orientation='h', title="Correlaciones Negativas con el Promedio",
                               color=negative_corr.values, color_continuous_scale="Reds",
                               labels={'x': 'Coeficiente de Correlación (absoluto)', 'y': 'Variable'})
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No se encontraron correlaciones negativas significativas")
        else:
            st.warning("No se pueden mostrar las correlaciones porque la columna 'promedio' no está disponible.")

        # Insights de correlaciones
        if 'promedio' in corr_matrix.columns and len(promedio_corr) > 0:
            strongest_corr = promedio_corr.iloc[0]
            strongest_var = promedio_corr.index[0]

            st.markdown(f"""
            <div class="insight-box">
                <h4>💡 Insights de Correlaciones</h4>
                <p>• <strong>Variable más correlacionada:</strong> {strongest_var} (r = {corr_matrix.loc[strongest_var, 'promedio']:.3f})</p>
                <p>• <strong>Correlaciones fuertes (|r| > 0.5):</strong> {len(promedio_corr[promedio_corr > 0.5])} variables</p>
                <p>• <strong>Correlaciones moderadas (0.3 < |r| < 0.5):</strong> {len(promedio_corr[(promedio_corr > 0.3) & (promedio_corr <= 0.5)])} variables</p>
                <p>• <strong>Correlaciones débiles (|r| < 0.3):</strong> {len(promedio_corr[promedio_corr <= 0.3])} variables</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ Datos de correlación no disponibles</h4>
                <p>No se pudo calcular la correlación con la variable 'promedio'. Verifica que el dataset contenga esta columna.</p>
            </div>
            """, unsafe_allow_html=True)

    # ==================== ANÁLISIS POR CATEGORÍAS ====================
    with tabs[3]:
        st.markdown("""
        <div class="tab-container">
            <h2>📈 Análisis por Variables Categóricas</h2>
            <p>Compara el rendimiento académico entre diferentes grupos y categorías de estudiantes.</p>
        </div>
        """, unsafe_allow_html=True)

        # Identificar variables categóricas
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        # Agregar variables numéricas discretas que puedan ser categóricas
        discrete_numeric = [col for col in df_clean.select_dtypes(include=[np.number]).columns
                           if df_clean[col].nunique() <= 10 and col != 'promedio']
        categorical_cols.extend(discrete_numeric)

        if len(categorical_cols) == 0:
            st.warning("No se encontraron variables categóricas en el dataset.")
        else:
            selected_cat = st.selectbox("Selecciona una variable categórica:", categorical_cols)

            # Tips para análisis categórico
            col1, col2 = st.columns(2)


            col1, col2 = st.columns(2)

            with col1:
                # Distribución de la variable categórica
                value_counts = df_clean[selected_cat].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Distribución de {selected_cat}",
                           color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Promedio por categoría
                avg_by_cat = df_clean.groupby(selected_cat)['promedio'].mean().sort_values(ascending=False)
                fig = px.bar(x=avg_by_cat.index, y=avg_by_cat.values,
                           title=f"Promedio Académico por {selected_cat}",
                           color=avg_by_cat.values,
                           color_continuous_scale="Viridis",
                           labels={'x': selected_cat, 'y': 'Promedio'})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Boxplot detallado
            fig = px.box(df_clean, x=selected_cat, y='promedio',
                        title=f"Distribución del Promedio por {selected_cat}",
                        color=selected_cat,
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Estadísticas por categoría
            st.subheader(f"📊 Estadísticas por {selected_cat}")
            stats_by_cat = df_clean.groupby(selected_cat)['promedio'].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(2)
            st.dataframe(stats_by_cat, use_container_width=True)

            # Test ANOVA si hay suficientes grupos
            if df_clean[selected_cat].nunique() >= 2:
                groups = [df_clean[df_clean[selected_cat] == cat]['promedio'].dropna()
                         for cat in df_clean[selected_cat].unique()]

                if len(groups) >= 2 and all(len(group) > 1 for group in groups):
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)

                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>🧪 Test ANOVA - Diferencias entre Grupos</h4>
                            <p><strong>Estadístico F:</strong> {f_stat:.4f}</p>
                            <p><strong>P-valor:</strong> {p_value:.4f}</p>
                            <p><strong>Interpretación:</strong> {"Existen diferencias significativas entre grupos (p < 0.05)" if p_value < 0.05 else "No existen diferencias significativas entre grupos (p ≥ 0.05)"}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except:
                        st.warning("No se pudo realizar el test ANOVA debido a la estructura de los datos")

    # ==================== ANÁLISIS DEL PROMEDIO ====================
    with tabs[4]:
        st.markdown("""
        <div class="tab-container">
            <h2>🎯 Análisis Detallado del Promedio Académico</h2>
            <p>Explora en profundidad la variable objetivo y su distribución en la población estudiantil.</p>
        </div>
        """, unsafe_allow_html=True)

        promedio_data = df_clean['promedio']

        # Métricas del promedio
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Promedio General", f"{promedio_data.mean():.2f}")
        with col2:
            st.metric("Mediana", f"{promedio_data.median():.2f}")
        with col3:
            st.metric("Desviación Estándar", f"{promedio_data.std():.2f}")
        with col4:
            st.metric("Rango", f"{promedio_data.max() - promedio_data.min():.2f}")


        # Distribución del promedio
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df_clean, x='promedio', nbins=20,
                             title="Distribución del Promedio Académico",
                             marginal="box",
                             color_discrete_sequence=[px.colors.qualitative.Set2[1]])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Categorización del promedio
            def categorize_promedio(score):
                if score >= 4.5:
                    return "Excelente (4.5-5.0)"
                elif score >= 4.0:
                    return "Bueno (4.0-4.49)"
                elif score >= 3.5:
                    return "Aceptable (3.5-3.99)"
                elif score >= 3.0:
                    return "Regular (3.0-3.49)"
                else:
                    return "Bajo (< 3.0)"

            df_clean['categoria_promedio'] = df_clean['promedio'].apply(categorize_promedio)
            cat_counts = df_clean['categoria_promedio'].value_counts()

            fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                        title="Categorización del Rendimiento Académico",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)

        # Análisis de percentiles
        st.subheader("📊 Análisis de Percentiles")
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        perc_values = [np.percentile(promedio_data, p) for p in percentiles]

        perc_df = pd.DataFrame({
            'Percentil': [f"P{p}" for p in percentiles],
            'Valor': perc_values,
            'Interpretación': [
                "Muy bajo", "Bajo", "Inferior", "Medio", "Superior", "Alto", "Muy alto"
            ]
        })

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(perc_df, x='Percentil', y='Valor',
                        title="Percentiles del Promedio Académico",
                        color='Valor', color_continuous_scale="Viridis",
                        labels={'Valor': 'Promedio'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(perc_df, use_container_width=True)



    # ==================== VALORES ATÍPICOS ====================
    with tabs[5]:
        st.markdown("""
        <div class="tab-container">
            <h2>🔍 Detección de Valores Atípicos</h2>
            <p>Identifica y analiza valores inusuales que pueden afectar el análisis y el modelo predictivo.</p>
        </div>
        """, unsafe_allow_html=True)

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        selected_var = st.selectbox("Selecciona una variable para analizar:", numeric_cols)


        # Método IQR para detectar outliers
        Q1 = df_clean[selected_var].quantile(0.25)
        Q3 = df_clean[selected_var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_clean[(df_clean[selected_var] < lower_bound) |
                           (df_clean[selected_var] > upper_bound)]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total de Outliers", len(outliers))
            st.metric("Porcentaje de Outliers", f"{(len(outliers)/len(df_clean)*100):.1f}%")

        with col2:
            st.metric("Límite Inferior", f"{lower_bound:.2f}")
            st.metric("Límite Superior", f"{upper_bound:.2f}")

        # Boxplot con outliers marcados
        fig = px.box(df_clean, y=selected_var,
                    title=f"Boxplot de {selected_var} - Detección de Outliers",
                    color_discrete_sequence=[px.colors.qualitative.Set2[2]])
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar outliers si existen
        if len(outliers) > 0:
            st.subheader("📋 Registros con Valores Atípicos")

            # Análisis adicional de outliers
            outlier_analysis = outliers[selected_var].describe()

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Promedio de outliers", f"{outlier_analysis['mean']:.2f}")
                st.metric("Outlier máximo", f"{outlier_analysis['max']:.2f}")

            with col2:
                st.metric("Desviación de outliers", f"{outlier_analysis['std']:.2f}")
                st.metric("Outlier mínimo", f"{outlier_analysis['min']:.2f}")

            st.dataframe(outliers, use_container_width=True)

            # Recomendaciones según el porcentaje de outliers
            outlier_percentage = (len(outliers) / len(df_clean)) * 100

            if outlier_percentage > 5:
                recommendation = "Considerar transformación de variables o métodos robustos"
                box_color = "warning"
            else:
                recommendation = "El porcentaje de outliers es manejable"
                box_color = "success"

            st.markdown(f"""
            <div class="insight-box">
                <h4>💡 Recomendaciones para Outliers</h4>
                <p><strong>Porcentaje de outliers:</strong> {outlier_percentage:.1f}%</p>
                <p><strong>Recomendación:</strong> {recommendation}</p>
                <p><strong>Opciones de manejo:</strong></p>
                <ul>
                    <li>Transformación logarítmica o de raíz cuadrada</li>
                    <li>Métodos de imputación con valores límite</li>
                    <li>Uso de algoritmos robustos a outliers</li>
                    <li>Eliminación solo si son errores de medición</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ No se detectaron valores atípicos en esta variable.")

except FileNotFoundError:
    st.markdown(f"""
    <div style='background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545;'>
        <h4>❌ Error: Dataset no encontrado</h4>
        <p>No se encontró el archivo 'dataset.xlsx'. Asegúrate de que esté en el directorio correcto y tenga el formato adecuado.</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.markdown(f"""
    <div style='background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545;'>
        <h4>❌ Error inesperado</h4>
        <p>Ocurrió un error al procesar los datos: {str(e)}</p>
        <p>Verifica que el archivo Excel tenga la estructura esperada y la hoja se llame 'Dataset'.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
    <p>Análisis Exploratorio de Datos (EDA) • Desarrollado con Streamlit y Python • 2023</p>
</div>
""", unsafe_allow_html=True)