import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1) 基础绘图风格与清晰度设置
try:
	import seaborn as sns

	sns.set_style("whitegrid")
except Exception:
	plt.style.use("default")

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def get_time_slice_indices(timestamps, start_date=None, days=2):
	"""根据起始日期与天数返回时间切片索引。"""
	if start_date is None:
		start_ts = timestamps.min().normalize()
	else:
		start_ts = pd.to_datetime(start_date)

	end_ts = start_ts + pd.Timedelta(days=days)
	mask = (timestamps >= start_ts) & (timestamps < end_ts)
	indices = np.where(mask)[0]

	if len(indices) == 0:
		raise ValueError(
			f"未找到时间范围内的数据: start_date={start_ts.date()}, days={days}"
		)

	return indices, start_ts


def main():
	parser = argparse.ArgumentParser(description="风电场 EDA 可视化（支持按天切片）")
	parser.add_argument(
		"--start-date",
		type=str,
		default=None,
		help="起始日期，例如 2012-03-01；不传则从数据首日开始",
	)
	parser.add_argument(
		"--days",
		type=int,
		default=2,
		help="绘图天数（建议 1 或 2）",
	)
	args = parser.parse_args()

	# 2) 创建目录结构
	os.makedirs("figures", exist_ok=True)
	os.makedirs("figures/power_curves", exist_ok=True)
	os.makedirs("figures/time_series", exist_ok=True)

	# 3) 数据加载与预处理
	data = np.load("data/wind_train_val_2012-01-02_to_2013-07-13.npy", allow_pickle=True)
	timestamps = pd.to_datetime(data[:, 0, 0])
	num_farms = data.shape[1]

	# 仅提取指定日期范围（默认2天）
	indices, start_ts = get_time_slice_indices(timestamps, args.start_date, args.days)
	timestamps_slice = timestamps[indices]
	suffix = f"{start_ts.strftime('%Y%m%d')}_{args.days}d"

	# 4) 循环绘制并保存图表
	for farm_id in range(10):
		if farm_id >= num_farms:
			break

		# 数据列顺序为 [timestamp, U10, V10, U100, V100, Power]
		Power = data[:, farm_id, 5].astype(float)[indices]
		U100 = data[:, farm_id, 3].astype(float)[indices]
		V100 = data[:, farm_id, 4].astype(float)[indices]
		WS100 = np.sqrt(U100**2 + V100**2)

		# 图表一：风速-功率散点图
		plt.figure(figsize=(8, 5))
		plt.scatter(WS100, Power, s=2, color="navy", alpha=0.2)
		plt.title(f"Wind Farm {farm_id} - Power Curve")
		plt.xlabel("Wind Speed 100m (m/s)")
		plt.ylabel("Power (MW)")
		plt.tight_layout()
		plt.savefig(f"figures/power_curves/farm_{farm_id}_power_curve_{suffix}.png")
		plt.close()

		# 图表二：切片区间的功率时序图
		plt.figure(figsize=(10, 5))
		plt.plot(timestamps_slice, Power, linewidth=1.5, color="#0F766E")
		plt.title(f"Wind Farm {farm_id} - Power Time Series ({suffix})")
		plt.xlabel("Time")
		plt.ylabel("Power (MW)")
		plt.xticks(rotation=45)
		plt.tight_layout()
		plt.savefig(f"figures/time_series/farm_{farm_id}_time_series_{suffix}.png")
		plt.close()

	# 5) 执行提示
	print(f"EDA 图表已成功生成并保存在 figures 目录下（起始日={start_ts.date()}，天数={args.days}）。")


if __name__ == "__main__":
	main()
