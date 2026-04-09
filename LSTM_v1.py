import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import argparse
from datetime import datetime


class WindDataset(Dataset):
	def __init__(self, data_path, mode="train", split_ratio=0.8, window_size=24, scaler=None):
		raw_data = np.load(data_path, allow_pickle=True)
		numeric_data = raw_data[:, :, 1:].astype(np.float32)

		ws100 = np.sqrt(numeric_data[:, :, 2] ** 2 + numeric_data[:, :, 3] ** 2)[:, :, np.newaxis]
		ws100_cube = ws100 ** 3

		# full_data: [numeric(5), ws100(1), ws100^3(1)] => 7 features
		self.full_data = np.concatenate([numeric_data, ws100, ws100_cube], axis=-1)

		self.power_idx = 4
		self.weather_idxs = [0, 1, 2, 3, 5, 6]

		train_len = int(len(self.full_data) * split_ratio)
		train_data_raw = self.full_data[:train_len]
		val_data_raw = self.full_data[train_len:]

		if mode == "train":
			self.scaler = MinMaxScaler()
			train_weather_flat = train_data_raw[:, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
			self.scaler.fit(train_weather_flat)
			self.raw_segment = train_data_raw
		else:
			if scaler is None:
				raise ValueError("Validation/Test mode requires scaler fitted on training split.")
			self.scaler = scaler
			self.raw_segment = val_data_raw

		self.data = self.raw_segment.copy()
		weather_part = self.raw_segment[:, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
		self.data[:, :, self.weather_idxs] = self.scaler.transform(weather_part).reshape(
			self.raw_segment.shape[0], self.raw_segment.shape[1], len(self.weather_idxs)
		)

		self.window_size = window_size
		self.indices = []
		for i in range(self.window_size, len(self.data) - self.window_size):
			for f in range(self.data.shape[1]):
				self.indices.append((i, f))

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		t, f = self.indices[idx]
		past_seq = self.data[t - self.window_size : t, f, :]  # (24, 7)
		future_weather = self.data[t : t + self.window_size, f, self.weather_idxs]  # (24, 6)
		x = np.concatenate([past_seq, future_weather], axis=-1)  # (24, 13)
		y = self.data[t : t + self.window_size, f, self.power_idx]  # (24,)
		return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class WindLSTM_Attention(nn.Module):
	def __init__(self, input_dim=13, hidden_dim=64, num_layers=1, output_dim=24, lstm_dropout=0.2, fc_dropout=0.3):
		super().__init__()
		# 与 baseline 一致：LSTM dropout 仅在 num_layers > 1 时生效
		effective_lstm_dropout = lstm_dropout if num_layers > 1 else 0.0
		self.lstm = nn.LSTM(
			input_dim,
			hidden_dim,
			num_layers,
			batch_first=True,
			dropout=effective_lstm_dropout,
		)
		self.attn_score = nn.Linear(hidden_dim, 1)
		self.head = nn.Sequential(
			nn.Linear(hidden_dim, 128),
			nn.ReLU(),
			nn.Dropout(fc_dropout),
			nn.Linear(128, output_dim),
		)

	def forward(self, x):
		out, _ = self.lstm(x)  # (B, T, H)
		scores = self.attn_score(out)  # (B, T, 1)
		attn_weights = F.softmax(scores, dim=1)
		context_vector = torch.sum(out * attn_weights, dim=1)  # (B, H)
		return self.head(context_vector)


def parse_args():
	parser = argparse.ArgumentParser(description="LSTM v3 (Attention + Physics Feature) 训练脚本")

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
	parser.add_argument("--input-dim", type=int, default=13)
	parser.add_argument("--hidden-dim", type=int, default=64)
	parser.add_argument("--num-layers", type=int, default=2)
	parser.add_argument("--output-dim", type=int, default=24)
	parser.add_argument("--lstm-dropout", type=float, default=0.2)
	parser.add_argument("--fc-dropout", type=float, default=0.3)

	return parser.parse_args()


def main():
	args = parse_args()

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	exp_dir = os.path.join(args.log_root, timestamp)
	os.makedirs(exp_dir, exist_ok=True)
	log_file = open(os.path.join(exp_dir, "train_log.txt"), "w")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_dataset = WindDataset(
		args.data_path,
		mode="train",
		split_ratio=args.split_ratio,
		window_size=args.window_size,
	)
	joblib.dump(train_dataset.scaler, os.path.join(exp_dir, "scaler.pkl"))

	val_dataset = WindDataset(
		args.data_path,
		mode="val",
		split_ratio=args.split_ratio,
		window_size=args.window_size,
		scaler=train_dataset.scaler,
	)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

	model = WindLSTM_Attention(
		input_dim=args.input_dim,
		hidden_dim=args.hidden_dim,
		num_layers=args.num_layers,
		output_dim=args.output_dim,
		lstm_dropout=args.lstm_dropout,
		fc_dropout=args.fc_dropout,
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	criterion = nn.MSELoss()
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=0.5,
		patience=5,
		verbose=True,
	)

	print(f"🚀 实验开始，保存至: {exp_dir}")
	for epoch in range(args.epochs):
		model.train()
		total_loss = 0.0
		for bx, by in train_loader:
			bx, by = bx.to(device), by.to(device)
			optimizer.zero_grad()
			pred = model(bx)
			loss = criterion(pred, by)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

		model.eval()
		val_mae = 0.0
		with torch.no_grad():
			for bx, by in val_loader:
				bx, by = bx.to(device), by.to(device)
				output = model(bx)
				val_mae += torch.mean(torch.abs(output - by)).item()

		train_loss_epoch = total_loss / len(train_loader)
		val_mae_epoch = val_mae / len(val_loader)
		scheduler.step(val_mae_epoch)

		status = f"Epoch {epoch + 1}, Loss: {train_loss_epoch:.4f}, Val MAE: {val_mae_epoch:.4f}"
		print(status)
		log_file.write(status + "\n")
		log_file.flush()

		if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
			save_path = os.path.join(exp_dir, f"lstm_epoch_{epoch + 1}.pth")
			torch.save(model.state_dict(), save_path)
			print(f"💾 模型已保存: {save_path}")

	log_file.close()


if __name__ == "__main__":
	main()
