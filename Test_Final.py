import os
import sys
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 动态将 Time-Series-Library 加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
tslib_path = os.path.join(current_dir, 'Time-Series-Library')
if tslib_path not in sys.path:
    sys.path.append(tslib_path)

from models import Informer 

def load_data_safe(file_path):
    """智能加载：自动探测并剥离时间戳，仅保留纯数值物理特征"""
    data = np.load(file_path, allow_pickle=True)
    
    if data.dtype == 'O' or data.dtype.char in ['S', 'U']:
        T, N, F = data.shape
        numeric_feature_idx = []
        
        # 探查这几个特征中，哪些是数字，哪一个是时间戳
        for f in range(F):
            sample_val = data[0, 0, f]
            try:
                # 尝试将它转换为浮点数
                float(sample_val)
                numeric_feature_idx.append(f)
            except (ValueError, TypeError):
                # 转换失败的，必然是时间戳字符串/对象
                pass
                
        print(f"      [!] 自动识别并剥离了时间戳！单节点特征数从 {F} 降维至纯数值的 {len(numeric_feature_idx)}")
        
        # 仅保留纯数值的特征切片
        data = data[:, :, numeric_feature_idx]
        
        # 剥离后安全转换为纯正的 float32 张量
        data = np.array(data.tolist(), dtype=np.float32)
        
    return data.astype(np.float32)

def predict_and_evaluate(train_npy_path, test_npy_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    
    print("==================================================")
    print("      Informer 风电多步预测独立评估程序启动       ")
    print("==================================================")

    # ---------------- 1. 数据加载与自适应维度解析 ----------------
    print(f"[*] 正在加载训练集以提取 Scaler 参数: {train_npy_path}")
    train_raw = load_data_safe(train_npy_path)
    if len(train_raw.shape) != 3:
        raise ValueError("训练集必须是 3D 张量 [Time, Nodes, Features]")
    
    T_train, N_nodes, F_features = train_raw.shape
    total_features = N_nodes * F_features
    # 自动推导 Power 所在的列索引 (例如: 0, 5, 10... 或 0, 6, 12...)
    power_idx = [i * F_features for i in range(N_nodes)] 
    
    print(f"    -> 探测到节点数: {N_nodes}, 单节点特征数: {F_features}, 总展平维度: {total_features}")
    print(f"    -> 功率特征索引为: {power_idx}")

    print(f"[*] 正在加载测试集: {test_npy_path}")
    test_raw = load_data_safe(test_npy_path)
    
    # ---------------- 2. 严格的归一化处理 ----------------
    print("[*] 正在进行无泄漏的 StandardScaler 归一化...")
    train_flatten = train_raw.reshape(T_train, -1)
    test_flatten = test_raw.reshape(test_raw.shape[0], -1)
    
    scaler = StandardScaler()
    scaler.fit(train_flatten) # 绝对只能在训练集上 fit!
    test_scaled = scaler.transform(test_flatten)
    
    # ---------------- 3. 模型构建与权重加载 ----------------
    print(f"[*] 正在初始化 Informer 并加载权重: {checkpoint_path}")
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
        distil = True  # <--- 修复报错的核心参数
        embed = 'timeF'
        freq = 'h'
        activation = 'gelu'
        output_attention = False
# 魔法函数兜底：如果底层源码还偷偷请求了别的参数（如 down_sampling 等），一律返回 None 防止报错
        def __getattr__(self, item):
            return None
            
    model = Informer.Model(Config()).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # ---------------- 4. 时空滑动窗口推理 (NWP 气象对齐) ----------------
    print("[*] 正在执行滑动窗口前向推理 (注入 NWP 预报，遮蔽未来功率)...")
    seq_len, label_len, pred_len = 24, 12, 24
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i in range(0, len(test_scaled) - seq_len - pred_len + 1):
            past_data = test_scaled[i : i + seq_len]
            future_real = test_scaled[i + seq_len : i + seq_len + pred_len].copy()
            
            # 【核心业务逻辑】：将未来 24H 的风功率强制清零
            future_real[:, power_idx] = 0.0
            
            batch_x = torch.tensor(past_data).unsqueeze(0).float().to(device)
            dec_inp = torch.tensor(np.concatenate([past_data[-label_len:], future_real], axis=0)).unsqueeze(0).float().to(device)
            
            # 时间特征占位符
            batch_x_mark = torch.zeros((1, seq_len, 4)).float().to(device)
            batch_y_mark = torch.zeros((1, label_len + pred_len, 4)).float().to(device)
            
            output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            predictions.append(output.cpu().numpy()[0])
            
            # 提取原始未归一化的真实值，用于最终算误差
            true_unscaled = test_flatten[i + seq_len : i + seq_len + pred_len, power_idx]
            ground_truths.append(true_unscaled)

    # ---------------- 5. 结果反归一化与指标计算 ----------------
    print("[*] 正在反归一化预测结果并计算风场评估指标...")
    predictions = np.array(predictions) # [Batches, 24, total_features]
    ground_truths = np.array(ground_truths) # [Batches, 24, Nodes]
    
    # 提取功率预测列
    pred_power_scaled = predictions[:, :, power_idx]
    
    # 反归一化
    means = scaler.mean_[power_idx]
    stds = scaler.scale_[power_idx]
    pred_power_real = pred_power_scaled * stds + means
    
    # ================= 保存预测张量 =================
    pred_save_path = os.path.join(ckpt_dir, 'final_predictions.npy')
    np.save(pred_save_path, pred_power_real)
    print(f"    [+] 预测张量已保存至: {pred_save_path}")
    
    # ================= 计算并保存风场指标 =================
    metrics_save_path = os.path.join(ckpt_dir, 'metrics_per_node.txt')
    avg_maes, avg_rmses = [], []
    
    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        f.write("========== 10个风场 Informer 独立评估报告 ==========\n")
        f.write(f"测试集样本量: {len(pred_power_real)} 个 24小时滑动窗口\n\n")
        
        for node_i in range(N_nodes):
            pred_node = pred_power_real[:, :, node_i].flatten()
            true_node = ground_truths[:, :, node_i].flatten()
            
            mae = np.mean(np.abs(pred_node - true_node))
            rmse = np.sqrt(np.mean((pred_node - true_node)**2))
            
            avg_maes.append(mae)
            avg_rmses.append(rmse)
            
            report_str = f"风场 Node_{node_i:02d} -> MAE: {mae:.4f} | RMSE: {rmse:.4f}"
            f.write(report_str + "\n")
            print("    " + report_str)
            
        global_mae = np.mean(avg_maes)
        global_rmse = np.mean(avg_rmses)
        f.write("\n================ 全局平均指标 ================\n")
        f.write(f"Global MAE: {global_mae:.4f}\n")
        f.write(f"Global RMSE: {global_rmse:.4f}\n")
        print("--------------------------------------------------")
        print(f"    ⭐ 全局 MAE : {global_mae:.4f}")
        print(f"    ⭐ 全局 RMSE: {global_rmse:.4f}")
        print("--------------------------------------------------")
        
    print(f"    [+] 评估报告已保存至: {metrics_save_path}")
    print("[*] 评估全流程圆满结束！")

if __name__ == "__main__":
    # ==================== 请核对并修改以下三个路径 ====================
    # 1. 你的训练集路径 (用于精准提取归一化参数)
    TRAIN_FILE = './data/wind_train_val_2012-01-02_to_2013-07-13.npy' 
    
    # 2. 你的测试集路径
    TEST_FILE = './data/wind_test_cleaned.npy' 
    
    # 3. 你训练出的最佳模型权重路径 (请确认这个文件夹名字是对的)
    CKPT_FILE = './Time-Series-Library/checkpoints/long_term_forecast_wind_informer_m2m_Informer_custom_ftM_sl24_ll12_pl24_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_wind_power_nwp_masked_decoder_0/checkpoint.pth' 
    # ==================================================================
    
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ 找不到训练集: {TRAIN_FILE}")
    elif not os.path.exists(TEST_FILE):
        print(f"❌ 找不到测试集: {TEST_FILE}")
    elif not os.path.exists(CKPT_FILE):
        print(f"❌ 找不到模型权重: {CKPT_FILE}")
    else:
        predict_and_evaluate(TRAIN_FILE, TEST_FILE, CKPT_FILE)