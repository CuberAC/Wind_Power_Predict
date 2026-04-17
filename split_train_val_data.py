import argparse
import os

import numpy as np


def split_and_save(data_path, train_output, val_output, train_ratio=0.8):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须在 (0, 1) 之间")

    data = np.load(data_path, allow_pickle=True)
    split_idx = int(data.shape[0] * train_ratio)

    if split_idx <= 0 or split_idx >= data.shape[0]:
        raise ValueError("切分比例不合法，导致训练集或验证集为空")

    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dir = os.path.dirname(train_output)
    val_dir = os.path.dirname(val_output)
    if train_dir:
        os.makedirs(train_dir, exist_ok=True)
    if val_dir:
        os.makedirs(val_dir, exist_ok=True)

    np.save(train_output, train_data)
    np.save(val_output, val_data)

    print("=" * 50)
    print(f"原始数据: {data_path}")
    print(f"原始形状: {data.shape}")
    print(f"训练集形状: {train_data.shape}")
    print(f"验证集形状: {val_data.shape}")
    print(f"训练集已保存到: {train_output}")
    print(f"验证集已保存到: {val_output}")
    print("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser(description="按时间顺序切分 wind_train_val 数据集")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/wind_train_val_2012-01-02_to_2013-07-13.npy",
        help="原始 npy 文件路径",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/wind_train_80.npy",
        help="训练集保存路径",
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default="data/wind_val_20.npy",
        help="验证集保存路径",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例，默认 0.8",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split_and_save(
        data_path=args.data_path,
        train_output=args.train_output,
        val_output=args.val_output,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
