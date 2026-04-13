import numpy as np
import pandas as pd
import os

def export_to_csv():
    # 1. 创建存放供 TSlib 读取的 CSV 文件夹
    out_dir = './Time-Series-Library/dataset/WindPower/'
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. 加载原始的合并数据集
    data_path = 'data/wind_train_val_2012-01-02_to_2013-07-13.npy'
    print(f"📦 正在加载原始数据: {data_path}")
    raw_data = np.load(data_path, allow_pickle=True)
    
    # 获取时间戳列 (第 0 列)
    timestamps = raw_data[:, 0, 0]
    
    # 获取纯数值列 (U10, V10, U100, V100, Power)
    numeric_data = raw_data[:, :, 1:].astype(np.float32)
    
    num_farms = numeric_data.shape[1]
    
    # 3. 循环 10 个风场，分别构造包含物理特征的 CSV
    print(f"🚀 正在为 TSlib 生成 {num_farms} 个标准 CSV 格式数据集...")
    for f in range(num_farms):
        # 取出当前风场的原始特征
        U10 = numeric_data[:, f, 0]
        V10 = numeric_data[:, f, 1]
        U100 = numeric_data[:, f, 2]
        V100 = numeric_data[:, f, 3]
        Power = numeric_data[:, f, 4]
        
        # 补全我们引以为傲的物理特征！
        WS10 = np.sqrt(U10**2 + V10**2)
        WS100 = np.sqrt(U100**2 + V100**2)
        WS100_Cube = WS100**3
        WDir_Rad = np.arctan2(V100, U100)
        Sin_WDir = np.sin(WDir_Rad)
        Cos_WDir = np.cos(WDir_Rad)
        
        # 组装成 DataFrame
        # TSlib 的默认习惯：第一列必须叫 'date'，其他列是特征，预测目标通常放最后一列 (OT) 或通过参数指定 (MS)。
        # 我们采用 'MS' (Multivariate to Single) 模式：用所有特征预测 Power。
        df = pd.DataFrame({
            'date': timestamps,
            'U10': U10, 'V10': V10, 'U100': U100, 'V100': V100,
            'WS10': WS10, 'WS100': WS100, 'WS100_Cube': WS100_Cube,
            'Sin_WDir': Sin_WDir, 'Cos_WDir': Cos_WDir,
            'Power': Power  # 🌟 将预测目标放在最后一列，方便统一配置
        })
        
        # 保存为 CSV
        csv_path = os.path.join(out_dir, f'Farm_{f}.csv')
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Farm {f} 保存完毕 -> {csv_path}")

    print("\n🎉 全部 CSV 转换完成！你的数据已经准备好迎接 Informer 的洗礼了！")

if __name__ == "__main__":
    export_to_csv()