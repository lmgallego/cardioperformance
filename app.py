"""
Cycling Performance Analysis Web App
A Streamlit application for analyzing cycling power and heart rate data
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
from cycling_analysis import (
    process_cycling_data,
    process_multiple_files,
    calculate_cp,
    inspect_raw_data
)

# Page configuration
st.set_page_config(
    page_title="Cycling Performance Analysis",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


def create_power_hr_plot(df):
    """Create power and heart rate time series plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Power plot
    if 'power_smooth' in df.columns:
        ax1.plot(df['time'] / 60, df['power_smooth'], 'b-', linewidth=1, label='Power (smoothed)')
        ax1.set_ylabel('Power (W)', fontsize=12)
        ax1.set_title('Power Output Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # Heart rate plot
    if 'heart_rate_smooth' in df.columns:
        ax2.plot(df['time'] / 60, df['heart_rate_smooth'], 'r-', linewidth=1, label='Heart Rate (smoothed)')
        ax2.set_ylabel('Heart Rate (bpm)', fontsize=12)
        ax2.set_xlabel('Time (minutes)', fontsize=12)
        ax2.set_title('Heart Rate Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    return fig


def create_quartile_analysis_plot(results):
    """Create quartile analysis bar charts"""
    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']

    # Extract rHRI values for each quartile
    rhri_values = []
    power_values = []

    for q in quartiles:
        rhri_key = f'avg_rHRI_increase_bpm_per_s_{q}'
        power_key = f'avg_power_increase_percent_cp_{q}'

        rhri_values.append(results.get(rhri_key, np.nan))
        power_values.append(results.get(power_key, np.nan))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # rHRI plot
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    bars1 = ax1.bar(quartiles, rhri_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('rHRI (bpm/s/W)', fontsize=12)
    ax1.set_xlabel('Power Quartile', fontsize=12)
    ax1.set_title('Cardiovascular Efficiency by Power Quartile\n(Lower is Better)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars1, rhri_values):
        if not np.isnan(val):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=10)

    # Power percentage plot
    bars2 = ax2.bar(quartiles, power_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Power (% of CP)', fontsize=12)
    ax2.set_xlabel('Power Quartile', fontsize=12)
    ax2.set_title('Average Power Output by Quartile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars2, power_values):
        if not np.isnan(val):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


def create_power_distribution_plot(df):
    """Create power distribution histogram"""
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'power_smooth' in df.columns:
        power_data = df['power_smooth'].dropna()
        ax.hist(power_data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Power (W)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Power Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics
        mean_power = power_data.mean()
        median_power = power_data.median()
        ax.axvline(mean_power, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_power:.1f} W')
        ax.axvline(median_power, color='green', linestyle='--', linewidth=2, label=f'Median: {median_power:.1f} W')
        ax.legend()

    plt.tight_layout()
    return fig


# Main header
st.markdown('<h1 class="main-header">🚴 Cycling Performance Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    analysis_mode = st.radio(
        "Analysis Mode",
        ["Single File", "Multiple Files"],
        help="Choose whether to analyze one file or multiple files"
    )

    st.markdown("---")

    st.subheader("📊 About")
    st.info("""
    This tool analyzes cycling performance data to calculate:
    - **Critical Power (CP)**: Sustainable power output
    - **W'**: Anaerobic work capacity
    - **rHRI**: Cardiovascular efficiency metric
    - **Quartile Analysis**: Performance by power zones
    """)

    st.markdown("---")

    st.subheader("📁 Data Format")
    st.write("""
    Upload CSV files with columns:
    - `time`: seconds
    - `watts` or `power`: watts
    - `heartrate` or `heart_rate`: bpm
    """)

# Main content
if analysis_mode == "Single File":
    st.header("📄 Single File Analysis")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=['csv'],
        help="Select a cycling data CSV file to analyze"
    )

    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with st.spinner("Processing file..."):
                # Read the uploaded file
                df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig')

                # Create a temporary file path
                temp_file = f"/tmp/{uploaded_file.name}"
                df_raw.to_csv(temp_file, index=False)

                # Process the data
                results, diagnostics = process_cycling_data(temp_file)

            st.success("✅ File processed successfully!")

            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Key Metrics",
                "📈 Visualizations",
                "🔍 Detailed Results",
                "⚠️ Data Quality"
            ])

            with tab1:
                st.subheader("Critical Power Analysis")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    cp_value = results.get('critical_power_W', 0)
                    st.metric(
                        label="Critical Power",
                        value=f"{cp_value:.1f} W",
                        help="Maximum sustainable power output"
                    )

                with col2:
                    w_prime = results.get('w_prime_J', 0)
                    st.metric(
                        label="W' (W-prime)",
                        value=f"{w_prime:.0f} J",
                        help="Anaerobic work capacity"
                    )

                with col3:
                    r_squared = results.get('r_squared', 0)
                    st.metric(
                        label="Model Fit (R²)",
                        value=f"{r_squared:.4f}",
                        help="Quality of CP model fit"
                    )

                with col4:
                    duration = results.get('duration_minutes', 0)
                    st.metric(
                        label="Duration",
                        value=f"{duration:.1f} min",
                        help="Total duration of the session"
                    )

                st.markdown("---")

                st.subheader("Cardiovascular Efficiency by Power Quartile")

                col1, col2, col3, col4 = st.columns(4)

                quartiles_data = []
                for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                    rhri_key = f'avg_rHRI_increase_bpm_per_s_{q}'
                    power_key = f'avg_power_increase_percent_cp_{q}'

                    rhri_val = results.get(rhri_key, np.nan)
                    power_val = results.get(power_key, np.nan)

                    with [col1, col2, col3, col4][i]:
                        st.metric(
                            label=f"{q} ({power_val:.1f}% CP)" if not np.isnan(power_val) else q,
                            value=f"{rhri_val:.6f}" if not np.isnan(rhri_val) else "N/A",
                            help=f"rHRI for quartile {q} (lower is better)"
                        )

            with tab2:
                st.subheader("Performance Visualizations")

                # Read the data again for plotting
                df_plot = pd.read_csv(temp_file, encoding='utf-8-sig')

                # Rename columns if needed
                if 'watts' in df_plot.columns:
                    df_plot = df_plot.rename(columns={'watts': 'power', 'heartrate': 'heart_rate'})

                # Process data for plotting (simplified version)
                df_plot.loc[(df_plot['power'] > 1800) | (df_plot['power'] < 0), 'power'] = np.nan
                df_plot.loc[(df_plot['heart_rate'] < 30) | (df_plot['heart_rate'] > 220), 'heart_rate'] = np.nan
                df_plot['power'] = df_plot['power'].interpolate(method='linear', limit_direction='both')
                df_plot['heart_rate'] = df_plot['heart_rate'].interpolate(method='linear', limit_direction='both')
                df_plot['power_smooth'] = df_plot['power'].rolling(window=30, center=True, min_periods=1).mean()
                df_plot['heart_rate_smooth'] = df_plot['heart_rate'].rolling(window=30, center=True, min_periods=1).mean()

                # Plot 1: Power and Heart Rate over time
                with st.expander("📈 Power and Heart Rate Over Time", expanded=True):
                    fig1 = create_power_hr_plot(df_plot)
                    st.pyplot(fig1)
                    plt.close(fig1)

                # Plot 2: Quartile Analysis
                with st.expander("📊 Quartile Analysis", expanded=True):
                    fig2 = create_quartile_analysis_plot(results)
                    st.pyplot(fig2)
                    plt.close(fig2)

                # Plot 3: Power Distribution
                with st.expander("📊 Power Distribution", expanded=True):
                    fig3 = create_power_distribution_plot(df_plot)
                    st.pyplot(fig3)
                    plt.close(fig3)

            with tab3:
                st.subheader("Detailed Analysis Results")

                # Convert results to DataFrame for display
                results_display = {k: [v] for k, v in results.items() if not k.startswith('avg_') and k not in ['file', 'status', 'position']}
                df_results = pd.DataFrame(results_display)

                st.dataframe(
                    df_results.T,
                    use_container_width=True,
                    height=400
                )

                st.markdown("---")

                st.subheader("Quartile Metrics")

                quartile_results = {k: v for k, v in results.items() if k.startswith('avg_')}
                if quartile_results:
                    df_quartiles = pd.DataFrame([quartile_results])
                    st.dataframe(
                        df_quartiles.T,
                        use_container_width=True,
                        height=400
                    )

            with tab4:
                st.subheader("Data Quality Report")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Total Rows",
                        value=f"{diagnostics['rows']:,}",
                        help="Total number of data points"
                    )

                    st.metric(
                        label="Power Outliers",
                        value=f"{diagnostics['power_outliers']}",
                        help="Values > 1800W or < 0W"
                    )

                    st.metric(
                        label="Missing Power (%)",
                        value=f"{diagnostics['na_power']:.2f}%",
                        help="Percentage of missing power data"
                    )

                with col2:
                    st.metric(
                        label="Power Range",
                        value=f"{diagnostics['power_range'][0]:.1f} - {diagnostics['power_range'][1]:.1f} W",
                        help="Min and max power values"
                    )

                    st.metric(
                        label="Heart Rate Outliers",
                        value=f"{diagnostics['heart_rate_outliers']}",
                        help="Values < 30 or > 220 bpm"
                    )

                    st.metric(
                        label="Missing Heart Rate (%)",
                        value=f"{diagnostics['na_heart_rate']:.2f}%",
                        help="Percentage of missing heart rate data"
                    )

                st.markdown("---")

                st.info("""
                **Data Quality Notes:**
                - Outliers are automatically removed and interpolated
                - A 30-second rolling mean is applied for smoothing
                - Missing values are interpolated linearly
                """)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Please upload a CSV file to begin analysis")

        # Show example data format
        with st.expander("📋 View Example Data Format"):
            example_data = pd.DataFrame({
                'time': [0, 1, 2, 3, 4],
                'power': [150, 160, 155, 165, 170],
                'heart_rate': [120, 125, 123, 128, 130]
            })
            st.dataframe(example_data, use_container_width=True)

else:  # Multiple Files
    st.header("📁 Multiple Files Analysis")

    uploaded_files = st.file_uploader(
        "Upload multiple CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Select multiple cycling data CSV files to analyze and compare"
    )

    if uploaded_files:
        try:
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
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

            st.success(f"✅ Successfully processed {len(uploaded_files)} files!")

            # Create tabs
            tab1, tab2, tab3 = st.tabs([
                "📊 Summary Statistics",
                "📈 Comparison",
                "📋 Detailed Results"
            ])

            with tab1:
                st.subheader("Overall Statistics")

                successful = (results_df['status'] == 'Success').sum()
                failed = len(results_df) - successful

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Files", len(results_df))
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", failed)

                st.markdown("---")

                # Calculate average metrics
                success_df = results_df[results_df['status'] == 'Success']

                if len(success_df) > 0:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        avg_cp = success_df['critical_power_W'].mean()
                        st.metric("Avg Critical Power", f"{avg_cp:.1f} W")

                    with col2:
                        avg_wp = success_df['w_prime_J'].mean()
                        st.metric("Avg W'", f"{avg_wp:.0f} J")

                    with col3:
                        avg_r2 = success_df['r_squared'].mean()
                        st.metric("Avg R²", f"{avg_r2:.4f}")

                    with col4:
                        avg_duration = success_df['duration_minutes'].mean()
                        st.metric("Avg Duration", f"{avg_duration:.1f} min")

            with tab2:
                st.subheader("File Comparison")

                if len(success_df) > 0:
                    # Create comparison plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                    # Critical Power comparison
                    files = success_df['file'].tolist()
                    cp_values = success_df['critical_power_W'].tolist()

                    ax1.barh(files, cp_values, color='steelblue', alpha=0.7)
                    ax1.set_xlabel('Critical Power (W)', fontsize=12)
                    ax1.set_title('Critical Power by File', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3, axis='x')

                    # W' comparison
                    wp_values = success_df['w_prime_J'].tolist()

                    ax2.barh(files, wp_values, color='coral', alpha=0.7)
                    ax2.set_xlabel("W' (J)", fontsize=12)
                    ax2.set_title("W' by File", fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='x')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown("---")

                    # Show comparison stats if available
                    if comparison_stats is not None:
                        st.subheader("Group Comparison (Top 5 vs Others)")
                        st.dataframe(
                            comparison_stats,
                            use_container_width=True
                        )
                else:
                    st.warning("No successful analyses to compare")

            with tab3:
                st.subheader("Detailed Results Table")

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    height=600
                )

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="cycling_analysis_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Please upload multiple CSV files to begin batch analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>🚴 Cycling Performance Analysis Tool | Built with Streamlit</p>
    <p>For questions or issues, refer to the documentation</p>
</div>
""", unsafe_allow_html=True)
