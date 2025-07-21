import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
import glob
from openpyxl import Workbook
import warnings
warnings.filterwarnings('ignore')

# Define sigmoid function
def sigmoid(x, A, B, C, D):
    return A + (B / (1 + np.exp(-C * (x - D))))

# Function to find sequences
def find_sequences(df, column, threshold, direction="increase"):
    sequences = []
    start = None
    
    for i in range(len(df)):
        value = df[column].iloc[i]
        
        if pd.isna(value):
            if start is not None:
                sequences.append((start, i))
                start = None
            continue
            
        if direction == "increase" and value > threshold:
            if start is None:
                start = i
        elif direction == "decrease" and value < -threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                sequences.append((start, i))
                start = None
    
    if start is not None:
        sequences.append((start, len(df)))
    
    return sequences

# Function to extract position from filename
def extract_position(filename):
    import re
    match = re.search(r'_N(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        return None

# Function to inspect raw data
def inspect_raw_data(data, file_name):
    summary = {
        'file': file_name,
        'rows': len(data),
        'na_power': (data['power'].isna().sum() / len(data)) * 100,
        'na_heart_rate': (data['heart_rate'].isna().sum() / len(data)) * 100,
        'power_outliers': ((data['power'] > 1800) | (data['power'] < 0)).sum(),
        'heart_rate_outliers': ((data['heart_rate'] < 30) | (data['heart_rate'] > 220)).sum(),
        'power_range': (data['power'].min(), data['power'].max()),
        'heart_rate_range': (data['heart_rate'].min(), data['heart_rate'].max())
    }
    
    print(f"Diagnostics for {file_name}:")
    print(f"  Rows = {summary['rows']}")
    print(f"  NA power = {summary['na_power']:.2f}%")
    print(f"  NA heart_rate = {summary['na_heart_rate']:.2f}%")
    print(f"  Outliers power (>1800 W) = {summary['power_outliers']}")
    print(f"  Outliers heart_rate (<30 or >220 bpm) = {summary['heart_rate_outliers']}")
    print(f"  Power range = [{summary['power_range'][0]:.2f}, {summary['power_range'][1]:.2f}]")
    print(f"  Heart_rate range = [{summary['heart_rate_range'][0]:.2f}, {summary['heart_rate_range'][1]:.2f}]")
    
    return summary

# Function to calculate critical power
def calculate_cp(data, durations=[60, 300, 720]):
    # Filter outliers
    data = data.copy()
    data.loc[(data['power'] > 1800) | (data['power'] < 0), 'power'] = np.nan
    
    # Interpolate missing values
    if data['power'].notna().sum() > 2:
        data['power'] = data['power'].interpolate(method='linear', limit_direction='both')
    
    max_powers = []
    
    for duration in durations:
        if len(data) >= duration:
            # Calculate rolling mean
            rolling_power = data['power'].rolling(window=duration, min_periods=1).mean()
            max_power = rolling_power.max()
            max_powers.append(max_power)
        else:
            max_powers.append(np.nan)
    
    # Linear regression for CP calculation
    valid_data = [(1/d, p) for d, p in zip(durations, max_powers) if not np.isnan(p)]
    
    if len(valid_data) >= 2:
        x_vals = np.array([x[0] for x in valid_data])
        y_vals = np.array([x[1] for x in valid_data])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        
        cp = intercept  # Critical Power
        w_prime = slope  # W'
        
        return {
            'critical_power_W': cp,
            'w_prime_J': w_prime,
            'r_squared': r_value**2,
            'p_value': p_value
        }
    else:
        return {
            'critical_power_W': np.nan,
            'w_prime_J': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan
        }

# Function to analyze by quartile
def analyze_by_quartile(df, sequences, column, quartile_col='power_quartile'):
    results = {}
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = []
        
        for start, end in sequences:
            seq_df = df.iloc[start:end]
            
            # Check if sequence has data in this quartile
            if quartile_col in seq_df.columns and q in seq_df[quartile_col].values:
                # Get values for this quartile
                q_values = seq_df[seq_df[quartile_col] == q][column].dropna()
                
                if len(q_values) > 0:
                    q_data.extend(q_values.tolist())
        
        if len(q_data) > 0:
            results[f'avg_{column}_{q}'] = np.mean(q_data)
            results[f'std_{column}_{q}'] = np.std(q_data)
            results[f'max_{column}_{q}'] = np.max(q_data)
            results[f'count_{q}'] = len(q_data)
        else:
            results[f'avg_{column}_{q}'] = np.nan
            results[f'std_{column}_{q}'] = np.nan
            results[f'max_{column}_{q}'] = np.nan
            results[f'count_{q}'] = 0
    
    return results

# Main processing function
def process_cycling_data(file_path):
    """
    Process a single cycling data file
    """
    # Read data
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # Rename columns if needed
    if 'watts' in df.columns:
        df = df.rename(columns={'watts': 'power', 'heartrate': 'heart_rate'})
    
    # Get filename
    filename = os.path.basename(file_path)
    
    # Inspect raw data
    diagnostics = inspect_raw_data(df, filename)
    
    # Clean outliers
    df.loc[(df['power'] > 1800) | (df['power'] < 0), 'power'] = np.nan
    df.loc[(df['heart_rate'] < 30) | (df['heart_rate'] > 220), 'heart_rate'] = np.nan
    
    # Interpolate missing values
    df['power'] = df['power'].interpolate(method='linear', limit_direction='both')
    df['heart_rate'] = df['heart_rate'].interpolate(method='linear', limit_direction='both')
    
    # Apply rolling mean smoothing
    df['power_smooth'] = df['power'].rolling(window=30, center=True, min_periods=1).mean()
    df['heart_rate_smooth'] = df['heart_rate'].rolling(window=30, center=True, min_periods=1).mean()
    
    # Calculate derivatives
    df['fc_deriv'] = df['heart_rate_smooth'].diff() / df['time'].diff()
    df['power_deriv'] = df['power_smooth'].diff() / df['time'].diff()
    
    # Calculate Critical Power
    cp_results = calculate_cp(df, durations=[60, 300, 720])
    cp = cp_results['critical_power_W']
    
    # Calculate rHRI (relative Heart Rate Increase)
    df['rHRI'] = np.nan
    mask = (df['power_smooth'] > 0) & (df['heart_rate_smooth'] > 0)
    df.loc[mask, 'rHRI'] = df.loc[mask, 'fc_deriv'] / df.loc[mask, 'power_smooth']
    
    # Calculate power as percentage of CP
    if not np.isnan(cp) and cp > 0:
        df['power_percent_cp'] = (df['power_smooth'] / cp) * 100
    else:
        df['power_percent_cp'] = np.nan
    
    # Define quartiles based on power percentage
    if df['power_percent_cp'].notna().sum() > 4:
        df['power_quartile'] = pd.qcut(df['power_percent_cp'].dropna(), 
                                       q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                                       duplicates='drop')
    
    # Find sequences for increases and decreases
    increase_sequences = find_sequences(df, 'fc_deriv', 0.1, direction="increase")
    decrease_sequences = find_sequences(df, 'fc_deriv', 0.1, direction="decrease")
    
    # Analyze metrics by quartile
    rHRI_results = analyze_by_quartile(df, increase_sequences, 'rHRI')
    fc_deriv_increase_results = analyze_by_quartile(df, increase_sequences, 'fc_deriv')
    fc_deriv_decrease_results = analyze_by_quartile(df, decrease_sequences, 'fc_deriv')
    power_percent_results = analyze_by_quartile(df, increase_sequences, 'power_percent_cp')
    hr_max_results = analyze_by_quartile(df, increase_sequences, 'heart_rate_smooth')
    
    # Create summary results
    summary_results = {
        'file': filename,
        'status': 'Success',
        'position': extract_position(filename),
        'total_rows': len(df),
        'duration_minutes': df['time'].max() / 60,
        **cp_results
    }
    
    # Add all quartile results
    for results_dict, prefix in [
        (rHRI_results, 'rHRI_increase_bpm_per_s'),
        (fc_deriv_increase_results, 'fc_deriv_increase_bpm_per_s'),
        (fc_deriv_decrease_results, 'fc_deriv_decrease_bpm_per_s'),
        (power_percent_results, 'power_increase_percent_cp'),
        (hr_max_results, 'fc_increase_bpm')
    ]:
        for key, value in results_dict.items():
            if 'avg_' in key:
                new_key = key.replace('avg_rHRI', f'avg_{prefix}')
                new_key = new_key.replace('avg_fc_deriv', f'avg_{prefix}')
                new_key = new_key.replace('avg_power_percent_cp', f'avg_{prefix}')
                new_key = new_key.replace('max_heart_rate_smooth', f'max_{prefix}')
                summary_results[new_key] = value
    
    return summary_results, diagnostics

# Function to process multiple files
def process_multiple_files(folder_path, pattern="*.csv"):
    """
    Process multiple cycling data files in a folder
    """
    # Find all matching files
    file_paths = glob.glob(os.path.join(folder_path, pattern))
    
    all_results = []
    all_diagnostics = []
    
    for file_path in file_paths:
        try:
            print(f"\nProcessing: {os.path.basename(file_path)}")
            results, diagnostics = process_cycling_data(file_path)
            all_results.append(results)
            all_diagnostics.append(diagnostics)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            all_results.append({
                'file': os.path.basename(file_path),
                'status': 'Error',
                'error': str(e)
            })
    
    # Convert to DataFrames
    results_df = pd.DataFrame(all_results)
    diagnostics_df = pd.DataFrame(all_diagnostics)
    
    # Create comparison between Top 5 and No Top 5
    if 'position' in results_df.columns:
        comparison_df = results_df[results_df['status'] == 'Success'].copy()
        comparison_df['group'] = comparison_df['position'].apply(
            lambda x: 'Top 5' if pd.notna(x) and x <= 5 else 'No Top 5'
        )
        
        # Group statistics
        comparison_stats = comparison_df.groupby('group').agg({
            'critical_power_W': ['mean', 'std'],
            'w_prime_J': ['mean', 'std'],
            'avg_rHRI_increase_bpm_per_s_Q1': ['mean', 'std'],
            'avg_rHRI_increase_bpm_per_s_Q2': ['mean', 'std'],
            'avg_rHRI_increase_bpm_per_s_Q3': ['mean', 'std'],
            'avg_rHRI_increase_bpm_per_s_Q4': ['mean', 'std']
        })
    else:
        comparison_stats = None
    
    return results_df, diagnostics_df, comparison_stats

# Example usage
if __name__ == "__main__":
    # Process single file
    file_path = "i88124106_streams.csv"
    if os.path.exists(file_path):
        results, diagnostics = process_cycling_data(file_path)
        print("\nProcessing complete!")
        print(f"Critical Power: {results['critical_power_W']:.2f} W")
        print(f"W': {results['w_prime_J']:.2f} J")
    
    # Process multiple files in a folder
    # folder_path = "./cycling_data"
    # results_df, diagnostics_df, comparison_stats = process_multiple_files(folder_path)
    # results_df.to_excel("cycling_analysis_results.xlsx", index=False)
