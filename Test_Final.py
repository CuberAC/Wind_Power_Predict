import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 将 Time-Series-Library 加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
tslib_path = os.path.join(current_dir, 'Time-Series-Library')
if tslib_path not in sys.path:
    sys.path.append(tslib_path)

from models import Informer 
from utils.timefeatures import time_features # 导入官方的时间特征生成器

def load_data_with_time(file_path):
    """智能加载：分离纯数值特征与真实时间戳"""
    data = np.load(file_path, allow_pickle=True)
    timestamps = None
    
    if data.dtype == 'O' or data.dtype.char in ['S', 'U']:
        T, N, F = data.shape
        numeric_feature_idx = []
        
        for f in range(F):
            sample_val = data[0, 0, f]
            try:
                float(sample_val)
                numeric_feature_idx.append(f)
            except:
                # 抓捕到时间戳列！(假设所有风场的时间戳一致，取 Node 0 的即可)
                timestamps = data[:, 0, f] 
                
        # 剥离出纯数值特征
        data = data[:, :, numeric_feature_idx]
        data = np.array(data.tolist(), dtype=np.float32)
        
    return data.astype(np.float32), timestamps

def predict_and_evaluate(train_npy_path, test_npy_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    
    # ---------------- 1. 加载数据与时间戳提取 ----------------
    train_raw, train_time = load_data_with_time(train_npy_path)
    test_raw, test_time = load_data_with_time(test_npy_path)
    
    T_train, N_nodes, F_features = train_raw.shape
    total_features = N_nodes * F_features
    power_idx = [i * F_features + 4 for i in range(N_nodes)]
    
    # 构建官方标准的时间特征 [T, 4]
    # 如果原数据没有提取到时间，使用一个假的时间序列兜底 (防止崩溃，但你的是有的)
    if test_time is None:
        test_time = pd.date_range(start='2013-07-14 00:00:00', periods=len(test_raw), freq='H')
    
    test_time_df = pd.to_datetime(test_time)
    test_time_marks = time_features(test_time_df, freq='h').transpose(1, 0)
    
    # ---------------- 2. 严格的归一化处理 (仅用前 70%) ----------------
    train_flatten = train_raw.reshape(T_train, -1)
    test_flatten = test_raw.reshape(test_raw.shape[0], -1)
    
    # TSlib 默认按 7:1:2 划分，Scaler 只在纯 train 上 fit
    num_train = int(T_train * 0.7) 
    pure_train_data = train_flatten[:num_train]
    
    scaler = StandardScaler()
    scaler.fit(pure_train_data) # [核心修复] 绝不能用整个 train_val 去 fit!
    test_scaled = scaler.transform(test_flatten)
    
    # ---------------- 3. 模型构建与权重加载 ----------------
    class Config:
        task_name = 'long_term_forecast'
        seq_len = 24
        label_len = 12
        pred_len = 24
        enc_in = total_features
        dec_in = total_features
        c_out = total_features
        d_model = 512
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 2048
        factor = 3
        dropout = 0.1
        embed = 'timeF'
        freq = 'h'
        activation = 'gelu'
        output_attention = False
        distil = True
        def __getattr__(self, item): return None

    model = Informer.Model(Config()).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # ---------------- 4. 时空滑动窗口推理 (注入 NWP & 真实时间) ----------------
    seq_len, label_len, pred_len = 24, 12, 24
    predictions, ground_truths = [], []
    
    with torch.no_grad():
        for i in range(0, len(test_scaled) - seq_len - pred_len + 1):
            past_data = test_scaled[i : i + seq_len]
            future_real = test_scaled[i + seq_len : i + seq_len + pred_len].copy()
            
            # 【核心逻辑】：遮蔽未来功率，保留 NWP
            future_real[:, power_idx] = 0.0
            
            batch_x = torch.tensor(past_data).unsqueeze(0).float().to(device)
            dec_inp = torch.tensor(np.concatenate([past_data[-label_len:], future_real], axis=0)).unsqueeze(0).float().to(device)
            
            # [核心修复]：注入真实的时间特征（切片对齐）
            b_x_mark = test_time_marks[i : i + seq_len]
            b_y_mark = test_time_marks[i + seq_len - label_len : i + seq_len + pred_len]
            
            batch_x_mark = torch.tensor(b_x_mark).unsqueeze(0).float().to(device)
            batch_y_mark = torch.tensor(b_y_mark).unsqueeze(0).float().to(device)
            
            output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # 取未来 pred_len 部分
            predictions.append(output[:, -pred_len:, :].cpu().numpy()[0]) 
            
            true_unscaled = test_flatten[i + seq_len : i + seq_len + pred_len, power_idx]
            ground_truths.append(true_unscaled)

    # ---------------- 5. 结果反归一化与指标计算 ----------------
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    pred_power_scaled = predictions[:, :, power_idx]
    true_power_scaled = np.array([
        test_scaled[i + seq_len : i + seq_len + pred_len, power_idx]
        for i in range(0, len(test_scaled) - seq_len - pred_len + 1)
    ])
    
    # 提取功率对应的 scaler 参数进行反推
    means = scaler.mean_[power_idx]
    stds = scaler.scale_[power_idx]
    pred_power_real = pred_power_scaled * stds + means

    # ================= 计算并保存风场指标 =================
    avg_maes, avg_rmses = [], []
    scaled_maes, scaled_mses = [], []
    print("\n" + "="*50)
    for node_i in range(N_nodes):
        pred_node = pred_power_real[:, :, node_i].flatten()
        true_node = ground_truths[:, :, node_i].flatten()

        pred_node_scaled = pred_power_scaled[:, :, node_i].flatten()
        true_node_scaled = true_power_scaled[:, :, node_i].flatten()

        if node_i == 0:
            print("[Probe] Node_00 true_node 分布:")
            print(f"    min={true_node.min():.6f}, max={true_node.max():.6f}, first5={true_node[:5]}")
            print("[Probe] Node_00 pred_node 分布:")
            print(f"    min={pred_node.min():.6f}, max={pred_node.max():.6f}, first5={pred_node[:5]}")
        
        mae = np.mean(np.abs(pred_node - true_node))
        rmse = np.sqrt(np.mean((pred_node - true_node)**2))

        mae_scaled = np.mean(np.abs(pred_node_scaled - true_node_scaled))
        mse_scaled = np.mean((pred_node_scaled - true_node_scaled)**2)

        avg_maes.append(mae), avg_rmses.append(rmse)
        scaled_maes.append(mae_scaled), scaled_mses.append(mse_scaled)
        print(f"    风场 Node_{node_i:02d} -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        
    global_mae, global_rmse = np.mean(avg_maes), np.mean(avg_rmses)
    global_mae_scaled, global_mse_scaled = np.mean(scaled_maes), np.mean(scaled_mses)
    print("-"*50)
    print(f"    ⭐ 全局 MAE : {global_mae:.4f}")
    print(f"    ⭐ 全局 RMSE: {global_rmse:.4f}")
    print(f"    ⭐ 全局 MAE (scaled): {global_mae_scaled:.4f}")
    print(f"    ⭐ 全局 MSE (scaled): {global_mse_scaled:.4f}")
    print("="*50)

if __name__ == "__main__":
    TRAIN_FILE = './data/wind_train_val_2012-01-02_to_2013-07-13.npy' 
    TEST_FILE = './data/wind_test_cleaned.npy' 
    CKPT_FILE = './Time-Series-Library/checkpoints/long_term_forecast_wind_informer_m2m_Informer_custom_ftM_sl24_ll12_pl24_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_wind_power_nwp_masked_decoder_0/checkpoint.pth' 
    predict_and_evaluate(TRAIN_FILE, TEST_FILE, CKPT_FILE)