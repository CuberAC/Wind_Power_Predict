# ==============================================================================
# 脚本名称: 06_dataset_informer.py
# 任务目标: 严格遵循 "24->24" 规则，为 Informer 框架构建标准化的 Dataset。
# 核心输出: batch_x (Encoder输入), batch_y (Decoder输入及预测目标)
# ==============================================================================

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class InformerWindDataset(Dataset):
    def __init__(self, data_path, mode='train', split_ratio=0.8, 
                 seq_len=24, label_len=12, pred_len=24, target_farm_id=0):
        """
        seq_len: Encoder 输入的过去序列长度 (严格遵守 24)
        label_len: Decoder 输入的已知引导序列长度 (取 12)
        pred_len: Decoder 需要预测的未来序列长度 (严格遵守 24)
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_farm_id = target_farm_id
        
        # 1. 基础数据加载与拆分 (复用你之前极其成熟的防泄露逻辑)
        raw_data = np.load(data_path, allow_pickle=True)
        numeric_data = raw_data[:, :, 1:].astype(np.float32)
        
        # 为了兼容开源框架，我们先把时间戳 (Index 0) 提出来，
        # Informer 内部自带强大的 TimeFeatures 嵌入层，不需要你手动算 Sin/Cos 了！
        # 这是 Informer 高级的地方，你需要保留时间戳字符串传给它。
        self.df_stamp = pd.DataFrame({'date': raw_data[:, 0, 0]})
        self.df_stamp['date'] = pd.to_datetime(self.df_stamp['date'])
        
        # 物理特征增强
        U10, V10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
        U100, V100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
        Power = numeric_data[:, :, 4]
        WS10 = np.sqrt(U10**2 + V10**2)
        WS100 = np.sqrt(U100**2 + V100**2)
        WS100_Cube = WS100 ** 3
        
        # 重新组合：[Power(必放第0列), U10, V10, U100, V100, WS10, WS100, Cube]
        # 注意：在多变量时序预测的标准库中，通常把预测目标 (Target) 放在第一列或者最后一列
        self.full_data = np.stack([Power, U10, V10, U100, V100, WS10, WS100, WS100_Cube], axis=-1)
        self.target_idx = 0 # Power 现在是第 0 列
        
        # 2. 严格按时间划分数据集，并执行单向标准化
        train_len = int(len(self.full_data) * split_ratio)
        if mode == 'train':
            self.data_x = self.full_data[:train_len, target_farm_id, :]
            self.data_stamp = self.df_stamp[:train_len]
        elif mode == 'val':
            # 注意：验证集的数据起点要稍微往前挪一点，为了能够切出第一个验证样本的历史 seq_len
            border1 = train_len - self.seq_len
            self.data_x = self.full_data[border1:, target_farm_id, :]
            self.data_stamp = self.df_stamp[border1:]
        elif mode == 'test':
            # 如果你有单独的测试集文件，逻辑类似，这里暂略
            pass
            
        # 🌟 极其严格的标准化防泄露
        self.scaler = StandardScaler()
        # 我们只拿训练集去 fit，且只 fit 气象特征 (索引 1 到 7)
        train_data_for_fit = self.full_data[:train_len, target_farm_id, :]
        self.scaler.fit(train_data_for_fit[:, 1:]) 
        
        # 转换气象特征，保留 Power (0-1) 不变
        self.data_x[:, 1:] = self.scaler.transform(self.data_x[:, 1:])
        
        # Informer 需要的时间特征矩阵 (后续结合官方库的 time_features 处理)
        # 这里为了简化，我们先抽出小时和月份作为示例
        self.data_stamp['month'] = self.data_stamp['date'].dt.month
        self.data_stamp['hour'] = self.data_stamp['date'].dt.hour
        self.data_stamp_vals = self.data_stamp[['month', 'hour']].values

    def __len__(self):
        # 样本数量 = 总长 - 历史长度 - 预测长度 + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # ================= 🌟 Informer 专属的数据切片逻辑 🌟 =================
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 1. Encoder 的输入 (过去 24h，包含 Power)
        seq_x = self.data_x[s_begin:s_end]
        
        # 2. Decoder 的输入 (过去 12h 包含 Power + 未来 24h 气象)
        seq_y = self.data_x[r_begin:r_end].copy() # 必须 copy，否则会修改原数组
        # ⚠️ 绝对防泄露：把未来 24 小时的 Power (第 0 列) 强制抹零为占位符！
        seq_y[-self.pred_len:, self.target_idx] = 0.0
        
        # 3. 真实的预测目标 (用于算 Loss)
        seq_y_true = self.data_x[r_begin:r_end]

        # 4. 对应的时间特征 (送入 TimeEmbedding)
        seq_x_mark = self.data_stamp_vals[s_begin:s_end]
        seq_y_mark = self.data_stamp_vals[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_true

# --- 简单测试一下 Dataloader ---
if __name__ == "__main__":
    dataset = InformerWindDataset('data/wind_train_val_2012-01-02_to_2013-07-13.npy')
    print(f"✅ Informer Dataset 构建成功！样本总数: {len(dataset)}")
    
    seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_true = dataset[0]
    print(f"Encoder 输入 seq_x shape: {seq_x.shape} (预期 24x8)")
    print(f"Decoder 输入 seq_y shape: {seq_y.shape} (预期 36x8, 其中最后24个点的Power必须为0)")
    
    # 验证防泄露
    print(f"\n🔍 抽查防泄露：Decoder 输入中未来最后 5 个小时的 Power 值：")
    print(seq_y[-5:, 0]) # 应该全都是 0.0