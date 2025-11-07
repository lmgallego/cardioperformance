"""
Cycling Performance Analysis Web App
A Streamlit application for analyzing cycling power and heart rate data
With internationalization support (English/Spanish) and interactive Plotly charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import sys
from cycling_analysis import (
    process_cycling_data,
    process_multiple_files,
    calculate_cp,
    inspect_raw_data
)

# Internationalization (i18n) - Translations
TRANSLATIONS = {
    'en': {
        'app_title': 'Cycling Performance Analysis',
        'app_description': 'Interactive cycling performance analysis tool',
        'language': 'Language',

        # Sidebar
        'configuration': 'Configuration',
        'analysis_mode': 'Analysis Mode',
        'single_file': 'Single File',
        'multiple_files': 'Multiple Files',
        'about': 'About',
        'about_text': '''
        This tool analyzes cycling performance data to calculate:
        - **Critical Power (CP)**: Sustainable power output
        - **W'**: Anaerobic work capacity
        - **rHRI**: Cardiovascular efficiency metric
        - **Quartile Analysis**: Performance by power zones
        ''',
        'data_format': 'Data Format',
        'data_format_text': '''
        Upload CSV files with columns:
        - `time`: seconds
        - `watts` or `power`: watts
        - `heartrate` or `heart_rate`: bpm
        ''',

        # Single file analysis
        'single_file_title': 'Single File Analysis',
        'upload_file': 'Upload a CSV file',
        'upload_file_help': 'Select a cycling data CSV file to analyze',
        'processing_file': 'Processing file...',
        'success_processed': 'File processed successfully!',

        # Tabs
        'tab_metrics': 'Key Metrics',
        'tab_visualizations': 'Visualizations',
        'tab_detailed': 'Detailed Results',
        'tab_quality': 'Data Quality',

        # Metrics
        'critical_power': 'Critical Power',
        'w_prime': "W' (W-prime)",
        'model_fit': 'Model Fit (R²)',
        'duration': 'Duration',
        'cp_help': 'Maximum sustainable power output',
        'wp_help': 'Anaerobic work capacity',
        'r2_help': 'Quality of CP model fit',
        'duration_help': 'Total duration of the session',

        # Quartile analysis
        'cv_efficiency': 'Cardiovascular Efficiency by Power Quartile',
        'cp_analysis': 'Critical Power Analysis',

        # Visualizations
        'performance_viz': 'Performance Visualizations',
        'power_hr_time': 'Power and Heart Rate Over Time',
        'quartile_analysis': 'Quartile Analysis',
        'power_distribution': 'Power Distribution',
        'power_output': 'Power Output',
        'heart_rate': 'Heart Rate',
        'time_minutes': 'Time (minutes)',
        'power_w': 'Power (W)',
        'hr_bpm': 'Heart Rate (bpm)',
        'power_quartile': 'Power Quartile',
        'rhri_label': 'rHRI (bpm/s/W)',
        'cv_efficiency_title': 'Cardiovascular Efficiency by Power Quartile',
        'cv_efficiency_subtitle': '(Lower is Better)',
        'avg_power_title': 'Average Power Output by Quartile',
        'power_percent_cp': 'Power (% of CP)',
        'frequency': 'Frequency',
        'mean': 'Mean',
        'median': 'Median',

        # Detailed results
        'detailed_results': 'Detailed Analysis Results',
        'quartile_metrics': 'Quartile Metrics',

        # Data quality
        'data_quality': 'Data Quality Report',
        'total_rows': 'Total Rows',
        'power_outliers': 'Power Outliers',
        'missing_power': 'Missing Power (%)',
        'power_range': 'Power Range',
        'hr_outliers': 'Heart Rate Outliers',
        'missing_hr': 'Missing Heart Rate (%)',
        'rows_help': 'Total number of data points',
        'power_outliers_help': 'Values > 1800W or < 0W',
        'hr_outliers_help': 'Values < 30 or > 220 bpm',
        'data_quality_notes': '''
        **Data Quality Notes:**
        - Outliers are automatically removed and interpolated
        - A 30-second rolling mean is applied for smoothing
        - Missing values are interpolated linearly
        ''',

        # Multiple files
        'multiple_files_title': 'Multiple Files Analysis',
        'upload_multiple': 'Upload multiple CSV files',
        'upload_multiple_help': 'Select multiple cycling data CSV files to analyze and compare',
        'processing_files': 'Processing {n} files...',
        'success_multiple': 'Successfully processed {n} files!',

        # Multiple files tabs
        'tab_summary': 'Summary Statistics',
        'tab_comparison': 'Comparison',
        'tab_results': 'Detailed Results',

        # Summary statistics
        'overall_stats': 'Overall Statistics',
        'total_files': 'Total Files',
        'successful': 'Successful',
        'failed': 'Failed',
        'avg_cp': 'Avg Critical Power',
        'avg_wp': "Avg W'",
        'avg_r2': 'Avg R²',
        'avg_duration': 'Avg Duration',

        # Comparison
        'file_comparison': 'File Comparison',
        'cp_by_file': 'Critical Power by File',
        'wp_by_file': "W' by File",
        'group_comparison': 'Group Comparison (Top 5 vs Others)',
        'no_comparison': 'No successful analyses to compare',

        # Results table
        'results_table': 'Detailed Results Table',
        'download_results': 'Download Results as CSV',

        # Info messages
        'upload_info': 'Please upload a CSV file to begin analysis',
        'example_format': 'View Example Data Format',
        'upload_multiple_info': 'Please upload multiple CSV files to begin batch analysis',

        # Errors
        'error_processing': 'Error processing file: {error}',
        'error_processing_multiple': 'Error processing files: {error}',

        # Footer
        'footer': '''
        Cycling Performance Analysis Tool | Built with Streamlit & Plotly
        For questions or issues, refer to the documentation
        ''',
    },
    'es': {
        'app_title': 'Análisis de Rendimiento Ciclista',
        'app_description': 'Herramienta interactiva de análisis de rendimiento ciclista',
        'language': 'Idioma',

        # Sidebar
        'configuration': 'Configuración',
        'analysis_mode': 'Modo de Análisis',
        'single_file': 'Archivo Único',
        'multiple_files': 'Múltiples Archivos',
        'about': 'Acerca de',
        'about_text': '''
        Esta herramienta analiza datos de rendimiento ciclista para calcular:
        - **Potencia Crítica (CP)**: Potencia sostenible
        - **W'**: Capacidad de trabajo anaeróbico
        - **rHRI**: Métrica de eficiencia cardiovascular
        - **Análisis por Cuartiles**: Rendimiento por zonas de potencia
        ''',
        'data_format': 'Formato de Datos',
        'data_format_text': '''
        Sube archivos CSV con las columnas:
        - `time`: segundos
        - `watts` o `power`: vatios
        - `heartrate` o `heart_rate`: ppm
        ''',

        # Single file analysis
        'single_file_title': 'Análisis de Archivo Único',
        'upload_file': 'Subir un archivo CSV',
        'upload_file_help': 'Selecciona un archivo CSV con datos de ciclismo para analizar',
        'processing_file': 'Procesando archivo...',
        'success_processed': '¡Archivo procesado exitosamente!',

        # Tabs
        'tab_metrics': 'Métricas Clave',
        'tab_visualizations': 'Visualizaciones',
        'tab_detailed': 'Resultados Detallados',
        'tab_quality': 'Calidad de Datos',

        # Metrics
        'critical_power': 'Potencia Crítica',
        'w_prime': "W' (W-prima)",
        'model_fit': 'Ajuste del Modelo (R²)',
        'duration': 'Duración',
        'cp_help': 'Potencia máxima sostenible',
        'wp_help': 'Capacidad de trabajo anaeróbico',
        'r2_help': 'Calidad del ajuste del modelo CP',
        'duration_help': 'Duración total de la sesión',

        # Quartile analysis
        'cv_efficiency': 'Eficiencia Cardiovascular por Cuartil de Potencia',
        'cp_analysis': 'Análisis de Potencia Crítica',

        # Visualizations
        'performance_viz': 'Visualizaciones de Rendimiento',
        'power_hr_time': 'Potencia y Frecuencia Cardíaca en el Tiempo',
        'quartile_analysis': 'Análisis por Cuartiles',
        'power_distribution': 'Distribución de Potencia',
        'power_output': 'Potencia de Salida',
        'heart_rate': 'Frecuencia Cardíaca',
        'time_minutes': 'Tiempo (minutos)',
        'power_w': 'Potencia (W)',
        'hr_bpm': 'Frecuencia Cardíaca (ppm)',
        'power_quartile': 'Cuartil de Potencia',
        'rhri_label': 'rHRI (ppm/s/W)',
        'cv_efficiency_title': 'Eficiencia Cardiovascular por Cuartil de Potencia',
        'cv_efficiency_subtitle': '(Menor es Mejor)',
        'avg_power_title': 'Potencia Promedio por Cuartil',
        'power_percent_cp': 'Potencia (% de CP)',
        'frequency': 'Frecuencia',
        'mean': 'Media',
        'median': 'Mediana',

        # Detailed results
        'detailed_results': 'Resultados de Análisis Detallados',
        'quartile_metrics': 'Métricas por Cuartiles',

        # Data quality
        'data_quality': 'Reporte de Calidad de Datos',
        'total_rows': 'Total de Filas',
        'power_outliers': 'Valores Atípicos de Potencia',
        'missing_power': 'Potencia Faltante (%)',
        'power_range': 'Rango de Potencia',
        'hr_outliers': 'Valores Atípicos de FC',
        'missing_hr': 'FC Faltante (%)',
        'rows_help': 'Número total de puntos de datos',
        'power_outliers_help': 'Valores > 1800W o < 0W',
        'hr_outliers_help': 'Valores < 30 o > 220 ppm',
        'data_quality_notes': '''
        **Notas sobre Calidad de Datos:**
        - Los valores atípicos se eliminan e interpolan automáticamente
        - Se aplica una media móvil de 30 segundos para suavizado
        - Los valores faltantes se interpolan linealmente
        ''',

        # Multiple files
        'multiple_files_title': 'Análisis de Múltiples Archivos',
        'upload_multiple': 'Subir múltiples archivos CSV',
        'upload_multiple_help': 'Selecciona múltiples archivos CSV de ciclismo para analizar y comparar',
        'processing_files': 'Procesando {n} archivos...',
        'success_multiple': '¡{n} archivos procesados exitosamente!',

        # Multiple files tabs
        'tab_summary': 'Estadísticas Resumen',
        'tab_comparison': 'Comparación',
        'tab_results': 'Resultados Detallados',

        # Summary statistics
        'overall_stats': 'Estadísticas Generales',
        'total_files': 'Total de Archivos',
        'successful': 'Exitosos',
        'failed': 'Fallidos',
        'avg_cp': 'CP Promedio',
        'avg_wp': "W' Promedio",
        'avg_r2': 'R² Promedio',
        'avg_duration': 'Duración Promedio',

        # Comparison
        'file_comparison': 'Comparación de Archivos',
        'cp_by_file': 'Potencia Crítica por Archivo',
        'wp_by_file': "W' por Archivo",
        'group_comparison': 'Comparación de Grupos (Top 5 vs Otros)',
        'no_comparison': 'No hay análisis exitosos para comparar',

        # Results table
        'results_table': 'Tabla de Resultados Detallados',
        'download_results': 'Descargar Resultados como CSV',

        # Info messages
        'upload_info': 'Por favor sube un archivo CSV para comenzar el análisis',
        'example_format': 'Ver Formato de Datos de Ejemplo',
        'upload_multiple_info': 'Por favor sube múltiples archivos CSV para comenzar el análisis por lotes',

        # Errors
        'error_processing': 'Error al procesar archivo: {error}',
        'error_processing_multiple': 'Error al procesar archivos: {error}',

        # Footer
        'footer': '''
        Herramienta de Análisis de Rendimiento Ciclista | Construida con Streamlit & Plotly
        Para preguntas o problemas, consulta la documentación
        ''',
    }
}


def t(key, **kwargs):
    """Translate key based on selected language"""
    lang = st.session_state.get('language', 'es')
    text = TRANSLATIONS.get(lang, TRANSLATIONS['es']).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


# Page configuration
st.set_page_config(
    page_title="Cycling Performance Analysis",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'es'

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


def create_power_hr_plot(df, lang='es'):
    """Create interactive power and heart rate time series plot with Plotly"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(t('power_output'), t('heart_rate'))
    )

    # Power plot
    if 'power_smooth' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'] / 60,
                y=df['power_smooth'],
                mode='lines',
                name=t('power_output'),
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>' + t('time_minutes') + '</b>: %{x:.1f}<br>' +
                              '<b>' + t('power_w') + '</b>: %{y:.1f}<extra></extra>'
            ),
            row=1, col=1
        )

    # Heart rate plot
    if 'heart_rate_smooth' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'] / 60,
                y=df['heart_rate_smooth'],
                mode='lines',
                name=t('heart_rate'),
                line=dict(color='#d62728', width=2),
                hovertemplate='<b>' + t('time_minutes') + '</b>: %{x:.1f}<br>' +
                              '<b>' + t('hr_bpm') + '</b>: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )

    # Update axes
    fig.update_xaxes(title_text=t('time_minutes'), row=2, col=1)
    fig.update_yaxes(title_text=t('power_w'), row=1, col=1)
    fig.update_yaxes(title_text=t('hr_bpm'), row=2, col=1)

    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_quartile_analysis_plot(results, lang='es'):
    """Create interactive quartile analysis bar charts with Plotly"""
    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']

    # Extract rHRI and power values
    rhri_values = []
    power_values = []

    for q in quartiles:
        rhri_key = f'avg_rHRI_increase_bpm_per_s_{q}'
        power_key = f'avg_power_increase_percent_cp_{q}'

        rhri_values.append(results.get(rhri_key, np.nan))
        power_values.append(results.get(power_key, np.nan))

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            t('cv_efficiency_title') + '<br>' + t('cv_efficiency_subtitle'),
            t('avg_power_title')
        ),
        horizontal_spacing=0.15
    )

    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

    # rHRI plot
    fig.add_trace(
        go.Bar(
            x=quartiles,
            y=rhri_values,
            marker=dict(color=colors, line=dict(color='black', width=1)),
            text=[f'{val:.6f}' if not np.isnan(val) else 'N/A' for val in rhri_values],
            textposition='outside',
            name='rHRI',
            hovertemplate='<b>' + t('power_quartile') + '</b>: %{x}<br>' +
                          '<b>' + t('rhri_label') + '</b>: %{y:.6f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Power percentage plot
    fig.add_trace(
        go.Bar(
            x=quartiles,
            y=power_values,
            marker=dict(color=colors, line=dict(color='black', width=1)),
            text=[f'{val:.1f}%' if not np.isnan(val) else 'N/A' for val in power_values],
            textposition='outside',
            name=t('power_percent_cp'),
            hovertemplate='<b>' + t('power_quartile') + '</b>: %{x}<br>' +
                          '<b>' + t('power_percent_cp') + '</b>: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )

    # Update axes
    fig.update_xaxes(title_text=t('power_quartile'), row=1, col=1)
    fig.update_xaxes(title_text=t('power_quartile'), row=1, col=2)
    fig.update_yaxes(title_text=t('rhri_label'), row=1, col=1)
    fig.update_yaxes(title_text=t('power_percent_cp'), row=1, col=2)

    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        template='plotly_white'
    )

    return fig


def create_power_distribution_plot(df, lang='es'):
    """Create interactive power distribution histogram with Plotly"""
    if 'power_smooth' not in df.columns:
        return None

    power_data = df['power_smooth'].dropna()
    mean_power = power_data.mean()
    median_power = power_data.median()

    # Create histogram
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=power_data,
            nbinsx=50,
            marker=dict(color='skyblue', line=dict(color='black', width=1)),
            name=t('power_distribution'),
            hovertemplate='<b>' + t('power_w') + '</b>: %{x:.1f}<br>' +
                          '<b>' + t('frequency') + '</b>: %{y}<extra></extra>'
        )
    )

    # Add mean line
    fig.add_vline(
        x=mean_power,
        line=dict(color='red', dash='dash', width=2),
        annotation_text=f"{t('mean')}: {mean_power:.1f} W",
        annotation_position="top"
    )

    # Add median line
    fig.add_vline(
        x=median_power,
        line=dict(color='green', dash='dash', width=2),
        annotation_text=f"{t('median')}: {median_power:.1f} W",
        annotation_position="bottom"
    )

    # Update layout
    fig.update_layout(
        title=t('power_distribution'),
        xaxis_title=t('power_w'),
        yaxis_title=t('frequency'),
        height=500,
        template='plotly_white',
        showlegend=False
    )

    return fig


def create_comparison_plots(success_df, lang='es'):
    """Create comparison plots for multiple files"""
    files = success_df['file'].tolist()
    cp_values = success_df['critical_power_W'].tolist()
    wp_values = success_df['w_prime_J'].tolist()

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(t('cp_by_file'), t('wp_by_file')),
        horizontal_spacing=0.15
    )

    # Critical Power comparison
    fig.add_trace(
        go.Bar(
            y=files,
            x=cp_values,
            orientation='h',
            marker=dict(color='steelblue', line=dict(color='black', width=1)),
            name=t('critical_power'),
            hovertemplate='<b>%{y}</b><br>' +
                          '<b>' + t('critical_power') + '</b>: %{x:.1f} W<extra></extra>'
        ),
        row=1, col=1
    )

    # W' comparison
    fig.add_trace(
        go.Bar(
            y=files,
            x=wp_values,
            orientation='h',
            marker=dict(color='coral', line=dict(color='black', width=1)),
            name=t('w_prime'),
            hovertemplate='<b>%{y}</b><br>' +
                          '<b>' + t('w_prime') + '</b>: %{x:.0f} J<extra></extra>'
        ),
        row=1, col=2
    )

    # Update axes
    fig.update_xaxes(title_text=t('power_w'), row=1, col=1)
    fig.update_xaxes(title_text='J', row=1, col=2)

    # Update layout
    fig.update_layout(
        height=max(400, len(files) * 50),
        showlegend=False,
        template='plotly_white'
    )

    return fig


# Main header
st.markdown(f'<h1 class="main-header">🚴 {t("app_title")}</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header(f"⚙️ {t('configuration')}")

    # Language selector
    language_options = {
        'Español': 'es',
        'English': 'en'
    }

    selected_lang = st.selectbox(
        f"🌐 {t('language')}",
        options=list(language_options.keys()),
        index=0 if st.session_state.language == 'es' else 1
    )

    # Update language in session state
    if language_options[selected_lang] != st.session_state.language:
        st.session_state.language = language_options[selected_lang]
        st.rerun()

    st.markdown("---")

    analysis_mode = st.radio(
        t('analysis_mode'),
        [t('single_file'), t('multiple_files')],
        help=t('analysis_mode')
    )

    st.markdown("---")

    st.subheader(f"📊 {t('about')}")
    st.info(t('about_text'))

    st.markdown("---")

    st.subheader(f"📁 {t('data_format')}")
    st.write(t('data_format_text'))

# Main content
if analysis_mode == t('single_file'):
    st.header(f"📄 {t('single_file_title')}")

    uploaded_file = st.file_uploader(
        t('upload_file'),
        type=['csv'],
        help=t('upload_file_help')
    )

    if uploaded_file is not None:
        try:
            with st.spinner(t('processing_file')):
                # Read the uploaded file
                df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig')

                # Create a temporary file path
                temp_file = f"/tmp/{uploaded_file.name}"
                df_raw.to_csv(temp_file, index=False)

                # Process the data
                results, diagnostics = process_cycling_data(temp_file)

            st.success(f"✅ {t('success_processed')}")

            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                f"📊 {t('tab_metrics')}",
                f"📈 {t('tab_visualizations')}",
                f"🔍 {t('tab_detailed')}",
                f"⚠️ {t('tab_quality')}"
            ])

            with tab1:
                st.subheader(t('cp_analysis'))

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    cp_value = results.get('critical_power_W', 0)
                    st.metric(
                        label=t('critical_power'),
                        value=f"{cp_value:.1f} W",
                        help=t('cp_help')
                    )

                with col2:
                    w_prime = results.get('w_prime_J', 0)
                    st.metric(
                        label=t('w_prime'),
                        value=f"{w_prime:.0f} J",
                        help=t('wp_help')
                    )

                with col3:
                    r_squared = results.get('r_squared', 0)
                    st.metric(
                        label=t('model_fit'),
                        value=f"{r_squared:.4f}",
                        help=t('r2_help')
                    )

                with col4:
                    duration = results.get('duration_minutes', 0)
                    st.metric(
                        label=t('duration'),
                        value=f"{duration:.1f} min",
                        help=t('duration_help')
                    )

                st.markdown("---")

                st.subheader(t('cv_efficiency'))

                col1, col2, col3, col4 = st.columns(4)

                for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                    rhri_key = f'avg_rHRI_increase_bpm_per_s_{q}'
                    power_key = f'avg_power_increase_percent_cp_{q}'

                    rhri_val = results.get(rhri_key, np.nan)
                    power_val = results.get(power_key, np.nan)

                    with [col1, col2, col3, col4][i]:
                        label = f"{q} ({power_val:.1f}% CP)" if not np.isnan(power_val) else q
                        value = f"{rhri_val:.6f}" if not np.isnan(rhri_val) else "N/A"
                        st.metric(
                            label=label,
                            value=value,
                            help=f"rHRI {q}"
                        )

            with tab2:
                st.subheader(t('performance_viz'))

                # Read and process data for plotting
                df_plot = pd.read_csv(temp_file, encoding='utf-8-sig')

                if 'watts' in df_plot.columns:
                    df_plot = df_plot.rename(columns={'watts': 'power', 'heartrate': 'heart_rate'})

                # Process data
                df_plot.loc[(df_plot['power'] > 1800) | (df_plot['power'] < 0), 'power'] = np.nan
                df_plot.loc[(df_plot['heart_rate'] < 30) | (df_plot['heart_rate'] > 220), 'heart_rate'] = np.nan
                df_plot['power'] = df_plot['power'].interpolate(method='linear', limit_direction='both')
                df_plot['heart_rate'] = df_plot['heart_rate'].interpolate(method='linear', limit_direction='both')
                df_plot['power_smooth'] = df_plot['power'].rolling(window=30, center=True, min_periods=1).mean()
                df_plot['heart_rate_smooth'] = df_plot['heart_rate'].rolling(window=30, center=True, min_periods=1).mean()

                # Plot 1: Power and Heart Rate over time
                with st.expander(f"📈 {t('power_hr_time')}", expanded=True):
                    fig1 = create_power_hr_plot(df_plot, st.session_state.language)
                    st.plotly_chart(fig1, use_container_width=True)

                # Plot 2: Quartile Analysis
                with st.expander(f"📊 {t('quartile_analysis')}", expanded=True):
                    fig2 = create_quartile_analysis_plot(results, st.session_state.language)
                    st.plotly_chart(fig2, use_container_width=True)

                # Plot 3: Power Distribution
                with st.expander(f"📊 {t('power_distribution')}", expanded=True):
                    fig3 = create_power_distribution_plot(df_plot, st.session_state.language)
                    if fig3:
                        st.plotly_chart(fig3, use_container_width=True)

            with tab3:
                st.subheader(t('detailed_results'))

                # Convert results to DataFrame for display
                results_display = {k: [v] for k, v in results.items()
                                 if not k.startswith('avg_') and k not in ['file', 'status', 'position']}
                df_results = pd.DataFrame(results_display)

                st.dataframe(
                    df_results.T,
                    use_container_width=True,
                    height=400
                )

                st.markdown("---")

                st.subheader(t('quartile_metrics'))

                quartile_results = {k: v for k, v in results.items() if k.startswith('avg_')}
                if quartile_results:
                    df_quartiles = pd.DataFrame([quartile_results])
                    st.dataframe(
                        df_quartiles.T,
                        use_container_width=True,
                        height=400
                    )

            with tab4:
                st.subheader(t('data_quality'))

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label=t('total_rows'),
                        value=f"{diagnostics['rows']:,}",
                        help=t('rows_help')
                    )

                    st.metric(
                        label=t('power_outliers'),
                        value=f"{diagnostics['power_outliers']}",
                        help=t('power_outliers_help')
                    )

                    st.metric(
                        label=t('missing_power'),
                        value=f"{diagnostics['na_power']:.2f}%",
                        help=t('rows_help')
                    )

                with col2:
                    st.metric(
                        label=t('power_range'),
                        value=f"{diagnostics['power_range'][0]:.1f} - {diagnostics['power_range'][1]:.1f} W",
                        help=t('power_range')
                    )

                    st.metric(
                        label=t('hr_outliers'),
                        value=f"{diagnostics['heart_rate_outliers']}",
                        help=t('hr_outliers_help')
                    )

                    st.metric(
                        label=t('missing_hr'),
                        value=f"{diagnostics['na_heart_rate']:.2f}%",
                        help=t('missing_hr')
                    )

                st.markdown("---")

                st.info(t('data_quality_notes'))

        except Exception as e:
            st.error(t('error_processing', error=str(e)))
            st.exception(e)

    else:
        st.info(f"👆 {t('upload_info')}")

        # Show example data format
        with st.expander(f"📋 {t('example_format')}"):
            example_data = pd.DataFrame({
                'time': [0, 1, 2, 3, 4],
                'power': [150, 160, 155, 165, 170],
                'heart_rate': [120, 125, 123, 128, 130]
            })
            st.dataframe(example_data, use_container_width=True)

else:  # Multiple Files
    st.header(f"📁 {t('multiple_files_title')}")

    uploaded_files = st.file_uploader(
        t('upload_multiple'),
        type=['csv'],
        accept_multiple_files=True,
        help=t('upload_multiple_help')
    )

    if uploaded_files:
        try:
            with st.spinner(t('processing_files', n=len(uploaded_files))):
                # Save all files temporarily
                import tempfile
                import os

                temp_dir = tempfile.mkdtemp()

                for uploaded_file in uploaded_files:
                    df_temp = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    df_temp.to_csv(temp_path, index=False)

                # Process all files
                results_df, diagnostics_df, comparison_stats = process_multiple_files(temp_dir)

            st.success(t('success_multiple', n=len(uploaded_files)))

            # Create tabs
            tab1, tab2, tab3 = st.tabs([
                f"📊 {t('tab_summary')}",
                f"📈 {t('tab_comparison')}",
                f"📋 {t('tab_results')}"
            ])

            with tab1:
                st.subheader(t('overall_stats'))

                successful = (results_df['status'] == 'Success').sum()
                failed = len(results_df) - successful

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(t('total_files'), len(results_df))
                with col2:
                    st.metric(t('successful'), successful)
                with col3:
                    st.metric(t('failed'), failed)

                st.markdown("---")

                # Calculate average metrics
                success_df = results_df[results_df['status'] == 'Success']

                if len(success_df) > 0:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        avg_cp = success_df['critical_power_W'].mean()
                        st.metric(t('avg_cp'), f"{avg_cp:.1f} W")

                    with col2:
                        avg_wp = success_df['w_prime_J'].mean()
                        st.metric(t('avg_wp'), f"{avg_wp:.0f} J")

                    with col3:
                        avg_r2 = success_df['r_squared'].mean()
                        st.metric(t('avg_r2'), f"{avg_r2:.4f}")

                    with col4:
                        avg_duration = success_df['duration_minutes'].mean()
                        st.metric(t('avg_duration'), f"{avg_duration:.1f} min")

            with tab2:
                st.subheader(t('file_comparison'))

                if len(success_df) > 0:
                    # Create comparison plot
                    fig = create_comparison_plots(success_df, st.session_state.language)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")

                    # Show comparison stats if available
                    if comparison_stats is not None:
                        st.subheader(t('group_comparison'))
                        st.dataframe(
                            comparison_stats,
                            use_container_width=True
                        )
                else:
                    st.warning(t('no_comparison'))

            with tab3:
                st.subheader(t('results_table'))

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    height=600
                )

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 {t('download_results')}",
                    data=csv,
                    file_name="cycling_analysis_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(t('error_processing_multiple', error=str(e)))
            st.exception(e)

    else:
        st.info(f"👆 {t('upload_multiple_info')}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>🚴 {t('footer')}</p>
</div>
""", unsafe_allow_html=True)
