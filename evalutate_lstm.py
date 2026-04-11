import torch
import numpy as np
import pandas as pd
import joblib
import os
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入上面定义的类 (或者确保在同一个目录下)
from LSTM_baseline import WindDataset, WindLSTM 

def evaluate_model(model_path):
    # 1. 基础检查
    exp_dir = os.path.dirname(model_path)
    scaler_path = os.path.join(exp_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print("❌ 错误：模型目录下没有找到 scaler.pkl")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载 Scaler 和 数据集 (验证集模式)
    scaler = joblib.load(scaler_path)
    dataset = WindDataset('data/wind_train_val_2012-01-02_to_2013-07-13.npy', mode='val', scaler=scaler)
    
    # 3. 加载模型
    model = WindLSTM().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 4. 按风场收集数据
    # 我们创建一个容器存 10 个风场的预测和真实值
    farm_results = {f: {'true': [], 'pred': []} for f in range(10)}
    
    # 使用无打乱的 DataLoader 遍历
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    print(f"🔍 正在评测模型: {model_path}...")
    with torch.no_grad():
        # 这里我们需要知道每个 batch 对应哪个风场
        # 回顾 Dataset.indices 的顺序是 (time_idx, farm_idx)
        current_idx = 0
        for bx, by in loader:
            output = model(bx.to(device)).cpu().numpy()
            target = by.numpy()
            
            # 把 batch 里的数据按索引归位
            for i in range(len(target)):
                _, farm_id = dataset.indices[current_idx]
                farm_results[farm_id]['true'].append(target[i])
                farm_results[farm_id]['pred'].append(output[i])
                current_idx += 1

    # 5. 计算每个风场的指标
    report_data = []
    for f in range(10):
        y_true = np.array(farm_results[f]['true'])
        y_pred = np.array(farm_results[f]['pred'])
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        report_data.append({
            "风场编号": f"Farm_{f}",
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })
        print(f"📍 Farm {f} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 6. 保存为 CSV
    df = pd.DataFrame(report_data)
    model_name = os.path.basename(model_path).replace(".pth", "")
    csv_path = os.path.join(exp_dir, f"eval_report_{model_name}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 评测报告已保存至: {csv_path}")
    print("\n全局平均表现:")
    print(df[["RMSE", "MAE"]].mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评测指定的 LSTM 模型")
    parser.add_argument("model_path", help="模型 .pth 文件路径")
    args = parser.parse_args()
    evaluate_model(args.model_path)