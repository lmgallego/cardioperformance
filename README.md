# Cycling Performance Analysis Tool

A Python implementation for analyzing cycling performance data, originally adapted from R code. This tool processes cycling power and heart rate data to calculate critical power, analyze cardiovascular efficiency, and generate comprehensive performance metrics.

## Features

- **Critical Power (CP) Analysis**: Calculates critical power and W' (anaerobic work capacity) using linear regression
- **Cardiovascular Efficiency Metrics**: Computes relative Heart Rate Increase (rHRI) to assess cardiovascular response to power output
- **Power Quartile Analysis**: Segments data into power quartiles for detailed performance analysis
- **Data Cleaning**: Handles outliers and missing values with interpolation
- **Visualization**: Generates comprehensive plots for performance analysis

## Requirements

```python
pandas
numpy
scipy
matplotlib
openpyxl
```

## Installation

```bash
pip install pandas numpy scipy matplotlib openpyxl
```

## Usage

### Basic Usage

```python
from cycling_analysis import process_cycling_data

# Process a single file
results, diagnostics = process_cycling_data('your_cycling_data.csv')

# Process multiple files
from cycling_analysis import process_multiple_files
results_df, diagnostics_df, comparison_stats = process_multiple_files('./cycling_data_folder')
```

### Input Data Format

The tool expects CSV files with the following columns:
- `time`: Time in seconds
- `watts` or `power`: Power output in watts
- `heartrate` or `heart_rate`: Heart rate in bpm

### Output

The analysis generates:
1. **Excel file** with comprehensive metrics
2. **Diagnostic information** about data quality
3. **Visualization plots** including:
   - Power and heart rate over time
   - rHRI by power quartile
   - Heart rate derivative by power quartile
   - Power distribution histogram

## Key Metrics Explained

### Critical Power (CP)
The maximum power output that can be sustained in a quasi-steady state without fatigue. Calculated using the relationship:
```
Power = CP + (W'/time)
```

### W' (W-prime)
The finite amount of work that can be done above critical power, representing anaerobic work capacity.

### rHRI (Relative Heart Rate Increase)
A novel metric that quantifies cardiovascular efficiency:
```
rHRI = (Heart Rate Derivative) / Power
```
Lower values indicate better cardiovascular efficiency.

## Algorithm Details

1. **Data Preprocessing**:
   - Remove outliers (power > 1800W or < 0W, HR < 30 or > 220 bpm)
   - Apply 30-second rolling mean smoothing
   - Interpolate missing values

2. **Critical Power Calculation**:
   - Extract maximum average power for 1, 5, and 12-minute durations
   - Apply linear regression to 1/time vs power relationship
   - Y-intercept = Critical Power, Slope = W'

3. **Quartile Analysis**:
   - Segment data based on power as percentage of CP
   - Calculate metrics for each quartile (Q1: lowest 25%, Q4: highest 25%)

4. **Sequence Detection**:
   - Identify periods of increasing/decreasing heart rate
   - Analyze cardiovascular response during these sequences

## Example Output

```
Critical Power: 208.69 W
W': 1835 J
Model fit (R²): 0.9056

rHRI by Power Quartile:
- Q1 (36.1% CP): 0.004244 bpm/s/W
- Q2 (57.5% CP): 0.001672 bpm/s/W
- Q3 (71.5% CP): 0.001255 bpm/s/W
- Q4 (96.8% CP): 0.001268 bpm/s/W
```

## Visualization Examples

The tool generates comprehensive visualizations including:
- Time series of power and heart rate
- Bar charts of metrics by power quartile
- Power distribution histograms
- Critical power curve fitting

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.

## Acknowledgments

Originally adapted from R code for cycling performance analysis. The Python implementation maintains the core analytical approach while leveraging Python's data science ecosystem.
