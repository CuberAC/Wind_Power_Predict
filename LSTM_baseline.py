import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse
from datetime import datetime

# --- Dataset 类保持之前的修正版 ---
class WindDataset(Dataset):
    def __init__(self, data_path, mode='train', split_ratio=0.8, window_size=24, scaler=None):
        raw_data = np.load(data_path, allow_pickle=True)
        numeric_data = raw_data[:, :, 1:].astype(np.float32)
        ws100 = np.sqrt(numeric_data[:, :, 2]**2 + numeric_data[:, :, 3]**2)[:, :, np.newaxis]
        self.full_data = np.concatenate([numeric_data, ws100], axis=-1)
        
        self.power_idx = 4
        self.weather_idxs = [0, 1, 2, 3, 5]
        
        train_len = int(len(self.full_data) * split_ratio)
        train_data_raw = self.full_data[:train_len]
        val_data_raw = self.full_data[train_len:]
        
        # 处理标准化器
        if mode == 'train':
            self.scaler = StandardScaler()
            train_weather_flat = train_data_raw[:, :, self.weather_idxs].reshape(-1, 5)
            self.scaler.fit(train_weather_flat)
            self.raw_segment = train_data_raw
        else:
            self.scaler = scaler # 验证集必须使用训练集的 scaler
            self.raw_segment = val_data_raw

        # 转换数据
        self.data = self.raw_segment.copy()
        weather_part = self.raw_segment[:, :, self.weather_idxs].reshape(-1, 5)
        self.data[:, :, self.weather_idxs] = self.scaler.transform(weather_part).reshape(self.raw_segment.shape[0], 10, 5)
            
        self.window_size = window_size
        self.indices = []
        for i in range(self.window_size, len(self.data) - self.window_size):
            for f in range(10):
                self.indices.append((i, f))

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        t, f = self.indices[idx]
        past_seq = self.data[t-self.window_size : t, f, :] 
        future_weather = self.data[t : t+self.window_size, f, self.weather_idxs]
        X = np.concatenate([past_seq, future_weather], axis=-1)
        Y = self.data[t : t+self.window_size, f, self.power_idx]
        return torch.tensor(X), torch.tensor(Y)

# --- 模型定义 ---
class WindLSTM(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, num_layers=2, output_dim=24, lstm_dropout=0.2, fc_dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(p=fc_dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # 获取最后一层的输出
        out = h_n[-1]
        
        # 在送入全连接层预测前执行 dropout
        out = self.dropout(out) 
        
        return self.fc(out)


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM baseline 训练脚本")

    # 数据相关参数
    parser.add_argument("--data-path", type=str, default="wind_train_val_2012-01-02_to_2013-07-13.npy")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--window-size", type=int, default=24)

    # 训练相关参数
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-root", type=str, default="logs")

    # 模型相关参数
    parser.add_argument("--input-dim", type=int, default=11)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--output-dim", type=int, default=24)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)
    parser.add_argument("--fc-dropout", type=float, default=0.3)

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 创建带时间戳的日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.log_root, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = open(os.path.join(exp_dir, "train_log.txt"), "w")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载数据并保存 Scaler
    train_dataset = WindDataset(
        args.data_path,
        mode='train',
        split_ratio=args.split_ratio,
        window_size=args.window_size,
    )
    joblib.dump(train_dataset.scaler, os.path.join(exp_dir, "scaler.pkl"))
    val_dataset = WindDataset(
        args.data_path,
        mode='val',
        split_ratio=args.split_ratio,
        window_size=args.window_size,
        scaler=train_dataset.scaler,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = WindLSTM(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        lstm_dropout=args.lstm_dropout,
        fc_dropout=args.fc_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    print(f"🚀 实验开始，保存至: {exp_dir}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        
        # 验证
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for bx, by in val_loader:
                output = model(bx.to(device))
                val_mae += torch.mean(torch.abs(output - by.to(device))).item()
        
        status = f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val MAE: {val_mae/len(val_loader):.4f}"
        print(status); log_file.write(status + "\n"); log_file.flush()

        # 按 save_every 保存模型
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(exp_dir, f"lstm_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 模型已保存: {save_path}")

    log_file.close()

if __name__ == "__main__": main()