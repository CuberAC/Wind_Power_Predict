import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def explore_data():
    file_path = 'data/wind_train_val_2012-01-02_to_2013-07-13.npy'
    data = np.load(file_path, allow_pickle=True)
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*50)
    print("1. Data cleaning and split")
    
    # 1. Extract timestamp column (same timestamp across farms, use Farm 0)
    # Convert to datetime for downstream time-based analysis
    timestamps = pd.to_datetime(data[:, 0, 0])
    
    # 2. Remove timestamp column, keep 5 numeric features
    numeric_data = data[:, :, 1:].astype(np.float32)
    
    print(f"Time length: {len(timestamps)} (from {timestamps[0]} to {timestamps[-1]})")
    print(f"Numeric data shape: {numeric_data.shape} -> (time, farm, weather/power features)")
    print("="*50)

    # Feature order: weather components + power
    feature_names = ['U10', 'V10', 'U100', 'V100', 'Power']

    # 2. Feature summary
    print("2. Feature statistics (Farm 0)")
    farm_0_data = numeric_data[:, 0, :]
    df_farm_0 = pd.DataFrame(farm_0_data, columns=feature_names)
    print(df_farm_0.describe().round(4))
    print("="*50)

    # 3. Correlation analysis
    fig_corr = plt.figure(figsize=(10, 8))
    # Synthetic wind speed magnitude from U/V components
    df_farm_0['WindSpeed_100m'] = np.sqrt(df_farm_0['U100']**2 + df_farm_0['V100']**2)
    
    corr_matrix = df_farm_0.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap (Farm 0)")
    corr_path = os.path.join(output_dir, 'corr_heatmap_farm0.png')
    fig_corr.savefig(corr_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {corr_path}")
    plt.show()

    # 4. Time-series plots
    plot_hours = 168 # 7 days (7 * 24)
    fig_ts = plt.figure(figsize=(15, 10))
    
    # Power curve
    plt.subplot(3, 1, 1)
    plt.plot(timestamps[:plot_hours], numeric_data[:plot_hours, 0, 4], label='Farm 0 - Power', color='red')
    plt.title('Farm 0 - Power (First 7 Days)')
    plt.legend()
    plt.grid(True)

    # Synthetic wind speed
    plt.subplot(3, 1, 2)
    wind_speed_100 = np.sqrt(numeric_data[:plot_hours, 0, 2]**2 + numeric_data[:plot_hours, 0, 3]**2)
    plt.plot(timestamps[:plot_hours], wind_speed_100, label='Farm 0 - 100m WindSpeed', color='blue', alpha=0.7)
    plt.title('Farm 0 - Wind Speed (First 7 Days)')
    plt.legend()
    plt.grid(True)
    
    # Spatial comparison: multiple farms
    plt.subplot(3, 1, 3)
    plt.plot(timestamps[:plot_hours], numeric_data[:plot_hours, 0, 4], label='Farm 0', alpha=0.8)
    plt.plot(timestamps[:plot_hours], numeric_data[:plot_hours, 1, 4], label='Farm 1', alpha=0.8)
    plt.plot(timestamps[:plot_hours], numeric_data[:plot_hours, 2, 4], label='Farm 2', alpha=0.8)
    plt.title('Farm Power Comparison (First 7 Days)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    ts_path = os.path.join(output_dir, 'timeseries_farm0_and_neighbors.png')
    fig_ts.savefig(ts_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {ts_path}")
    plt.show()


if __name__ == "__main__":
    explore_data()