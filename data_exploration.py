import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------- 配置部分 ----------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体，Mac用 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False
# ------------------------------------------

def explore_data():
    file_path = 'wind_train_val_2012-01-02_to_2013-07-13.npy'
    data = np.load(file_path, allow_pickle=True)
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*50)
    print("1. 数据清洗与分离")
    
    # 1. 提取时间列（由于10个风电场在同一时刻的时间是一样的，取 Farm 0 的时间即可）
    # pd.to_datetime 可以把字符串变成标准的时间格式，方便以后按天、月提取特征
    timestamps = pd.to_datetime(data[:, 0, 0])
    
    # 2. 剥离时间列，保留后面的5个纯数值特征，并强制转换为 float32
    numeric_data = data[:, :, 1:].astype(np.float32)
    
    print(f"时间序列长度: {len(timestamps)} (从 {timestamps[0]} 到 {timestamps[-1]})")
    print(f"纯数值数据维度: {numeric_data.shape} -> (时间点, 风电场, 气象与出力特征)")
    print("="*50)

    # 我们约定列的含义：0,1,2,3 是气象预报(U/V风速)，4 是风电出力
    feature_names = ['U10 (风速分量)', 'V10 (风速分量)', 'U100 (高空风速)', 'V100 (高空风速)', 'Power (风电出力)']

    # 2. 破解特征身份
    print("2. 特征统计信息 (以 Farm 0 为例)")
    farm_0_data = numeric_data[:, 0, :]
    df_farm_0 = pd.DataFrame(farm_0_data, columns=feature_names)
    print(df_farm_0.describe().round(4))
    print("="*50)

    # 3. 相关性分析
    fig_corr = plt.figure(figsize=(10, 8))
    # 计算一个合成风速（风速大小等于U和V的平方和开根号），看看合成风速与出力的相关性
    df_farm_0['WindSpeed_100m_合成'] = np.sqrt(df_farm_0['U100 (高空风速)']**2 + df_farm_0['V100 (高空风速)']**2)
    
    corr_matrix = df_farm_0.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("单一风电场内不同特征的相关系数热力图 (重点看出力和谁相关最高)")
    corr_path = os.path.join(output_dir, 'corr_heatmap_farm0.png')
    fig_corr.savefig(corr_path, dpi=300, bbox_inches='tight')
    print(f"已保存图像: {corr_path}")
    plt.show()

    # 4. 时序可视化
    plot_hours = 168 # 画7天的数据 (7 * 24)
    fig_ts = plt.figure(figsize=(15, 10))
    
    # 画出出力曲线 (第4列)
    plt.subplot(3, 1, 1)
    plt.plot(timestamps[:plot_hours], numeric_data[:plot_hours, 0, 4], label='Farm 0 - 出力 (Power)', color='red')
    plt.title('风电场 0 - 前7天风电出力曲线')
    plt.legend()
    plt.grid(True)

    # 画合成风速
    plt.subplot(3, 1, 2)
    wind_speed_100 = np.sqrt(numeric_data[:plot_hours, 0, 2]**2 + numeric_data[:plot_hours, 0, 3]**2)
    plt.plot(timestamps[:plot_hours], wind_speed_100, label='Farm 0 - 100m高空合成风速', color='blue', alpha=0.7)
    plt.title('风电场 0 - 对应时段的风速变化 (仔细观察风速高时，出力是否也高)')
    plt.legend()
    plt.grid(True)
    
    # 画空间相关性：10个风电场的出力对比
    plt.subplot(3, 1, 3)
    plt.plot(timestamps[:plot_hours], numeric_data[:plot_hours, 0, 4], label='Farm 0', alpha=0.8)
    plt.plot(timestamps[:plot_hours], numeric_data[:plot_hours, 1, 4], label='Farm 1', alpha=0.8)
    plt.plot(timestamps[:plot_hours], numeric_data[:plot_hours, 2, 4], label='Farm 2', alpha=0.8)
    plt.title('多个相近风电场 - 前7天出力曲线对比 (如果重合度极高，说明它们地理位置极其靠近)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    ts_path = os.path.join(output_dir, 'timeseries_farm0_and_neighbors.png')
    fig_ts.savefig(ts_path, dpi=300, bbox_inches='tight')
    print(f"已保存图像: {ts_path}")
    plt.show()


if __name__ == "__main__":
    explore_data()