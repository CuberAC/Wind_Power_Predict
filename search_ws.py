import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_power_curve(data_path="data/wind_train_val_2012-01-02_to_2013-07-13.npy", node_idx=0):
    """
    分析风速与功率的物理关系，寻找切入、满发、切出风速。
    默认分析第 0 个风电场 (node_idx=0)，如果你想看其他风场可以修改参数。
    """
    print(f"🚀 正在加载数据并分析风场 {node_idx} 的物理特性...")
    raw_data = np.load(data_path, allow_pickle=True)
    
    # 提取 U10(1), V10(2), Power(5)
    u10 = raw_data[:, node_idx, 1].astype(float)
    v10 = raw_data[:, node_idx, 2].astype(float)
    power = raw_data[:, node_idx, 5].astype(float)
    
    # 计算 10m 绝对风速 (你也可以换成 100m 风速来画图)
    ws = np.sqrt(u10**2 + v10**2)
    
    # 组合成 DataFrame 方便统计
    df = pd.DataFrame({'WS': ws, 'Power': power})
    
    # 去除 NaN 
    df = df.dropna()
    
    # ========================================================
    # 统计 1：寻找“切入风速 (Cut-in)”
    # 逻辑：风速在什么范围内，功率绝大多数时间都是 0？
    # ========================================================
    zero_power_df = df[df['Power'] < 0.01] # 考虑到噪声，设定一个小阈值
    if not zero_power_df.empty:
        # 排除掉那些风速极大但因为故障/切出而停机的情况，只看低风速区间
        low_wind_zeros = zero_power_df[zero_power_df['WS'] < 8.0]
        # 取 95% 分位数作为切入风速的粗略估计（因为偶尔会有低风速但依然有微弱功率的噪声）
        estimated_cut_in = low_wind_zeros['WS'].quantile(0.95)
        print(f"\n🌬️ 【估计的切入风速 (Cut-in)】: 大约在 {estimated_cut_in:.2f} m/s 附近")
        print(f"   (解读: 风速低于此值时，功率极大概率为 0)")
    else:
        print("\n🌬️ 未找到明显的零功率区间。")

    # ========================================================
    # 统计 2：寻找“满发风速 (Rated)”
    # 逻辑：风速大于多少时，功率达到最大值（如 1.0 或 0.99）且不再增加？
    # ========================================================
    full_power_df = df[df['Power'] > 0.95] # 考虑到归一化和波动，设定 0.95 为满发阈值
    if not full_power_df.empty:
        # 取 5% 分位数作为满发风速的下限（刚碰到满发平台的那个点）
        estimated_rated = full_power_df['WS'].quantile(0.05)
        print(f"\n🌪️ 【估计的额定/满发风速 (Rated)】: 大约在 {estimated_rated:.2f} m/s 附近")
        print(f"   (解读: 风速高于此值时，功率极大概率卡在满发状态)")
    else:
        print("\n🌪️ 未找到明显的满发区间。")
        
    # ========================================================
    # 统计 3：寻找“切出风速 (Cut-out)”
    # 逻辑：风速极大，但功率却掉到了 0。
    # ========================================================
    high_wind_zeros = zero_power_df[zero_power_df['WS'] > 15.0]
    if not high_wind_zeros.empty:
        estimated_cut_out = high_wind_zeros['WS'].min()
        print(f"\n🛑 【估计的切出/强风停机风速 (Cut-out)】: 大约在 {estimated_cut_out:.2f} m/s 附近")
    else:
        print(f"\n🛑 未找到明显的切出停机现象 (可能数据集中未出现极端大风，最高风速为 {df['WS'].max():.2f} m/s)。")

    # ========================================================
    # 画图：直观感受“风功率曲线 (Power Curve)”
    # ========================================================
    print("\n📈 正在生成风功率散点图，请查看弹出的窗口 (或保存的图片)...")
    plt.figure(figsize=(10, 6))
    
    # 画散点图，设置透明度 alpha 方便看清密集区域
    plt.scatter(df['WS'], df['Power'], alpha=0.1, s=2, color='blue')
    
    plt.title(f"Wind Power Curve (Node {node_idx})", fontsize=14)
    plt.xlabel("Wind Speed (m/s)", fontsize=12)
    plt.ylabel("Normalized Power", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 划出估计的阈值线
    if not zero_power_df.empty and 'estimated_cut_in' in locals():
        plt.axvline(x=estimated_cut_in, color='red', linestyle='--', label=f'Est. Cut-in: {estimated_cut_in:.1f}')
    if not full_power_df.empty and 'estimated_rated' in locals():
        plt.axvline(x=estimated_rated, color='green', linestyle='--', label=f'Est. Rated: {estimated_rated:.1f}')
    if not high_wind_zeros.empty and 'estimated_cut_out' in locals():
        plt.axvline(x=estimated_cut_out, color='purple', linestyle='--', label=f'Est. Cut-out: {estimated_cut_out:.1f}')
        
    plt.legend()
    plt.tight_layout()
    
    # 如果你在服务器上跑没有图形界面，请注释掉 plt.show()，使用 plt.savefig()
    # plt.savefig(f"power_curve_node_{node_idx}.png")
    plt.show()

if __name__ == "__main__":
    # 你可以循环 10 个节点，看看大家的风机型号（阈值）是不是一样的
    analyze_power_curve()