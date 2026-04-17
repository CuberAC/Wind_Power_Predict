import argparse
import os

import joblib
import numpy as np


def add_physical_features(ws_array):
    """基于风机物理边界生成硬截断与布尔开关特征。"""
    cut_in = 3.71
    rated = 4.50
    cut_out = 15.0

    ws = np.asarray(ws_array, dtype=np.float32)

    is_zero_power = ((ws < cut_in) | (ws > cut_out)).astype(np.float32)
    is_full_power = ((ws >= rated) & (ws <= cut_out)).astype(np.float32)
    is_ramp_zone = ((ws >= cut_in) & (ws < rated)).astype(np.float32)

    ws_clipped = np.where(
        (ws < cut_in) | (ws > cut_out),
        0.0,
        np.where(ws >= rated, rated, ws),
    ).astype(np.float32)

    return is_zero_power, is_full_power, is_ramp_zone, ws_clipped


def build_ridge_infer_features(raw_data, horizon=24):
    """只构建推理特征，不使用未来出力标签。"""
    if raw_data.ndim != 3:
        raise ValueError(f"raw_data 应为 3 维 [T, N, F]，实际为 {raw_data.shape}")
    if raw_data.shape[-1] < 5:
        raise ValueError("raw_data 最后一维至少应包含 time,u10,v10,u100,v100")

    raw_data = raw_data.astype(object)
    t_steps, num_nodes, _ = raw_data.shape

    if t_steps < (3 + horizon):
        raise ValueError(f"时间步不足，至少需要 {3 + horizon}，当前 {t_steps}")

    u100 = raw_data[:, :, 3].astype(np.float32)
    v100 = raw_data[:, :, 4].astype(np.float32)

    ws = np.sqrt(u100 ** 2 + v100 ** 2).astype(np.float32)
    wd = np.arctan2(v100, u100).astype(np.float32)
    sin_wd = np.sin(wd).astype(np.float32)
    cos_wd = np.cos(wd).astype(np.float32)
    is_zero_power, is_full_power, is_ramp_zone, ws_clipped = add_physical_features(ws)

    x_rows = []

    for t in range(2, t_steps - horizon):
        ws_t_all = ws[t]
        for n in range(num_nodes):
            hist_ws = ws[t - 2 : t + 1, n]
            hist_mean = float(hist_ws.mean())
            hist_max = float(hist_ws.max())
            first_diff = float(ws[t, n] - ws[t - 1, n])

            ws_curr = float(ws[t, n])
            ws_curr_cube = ws_curr ** 3
            sin_curr = float(sin_wd[t, n])
            cos_curr = float(cos_wd[t, n])
            zero_curr = float(is_zero_power[t, n])
            full_curr = float(is_full_power[t, n])
            ramp_curr = float(is_ramp_zone[t, n])
            ws_clip_curr = float(ws_clipped[t, n])
            ws_clip_curr_cube = ws_clip_curr ** 3

            others_ws = np.delete(ws_t_all, n).astype(np.float32)

            fut_ws = ws[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_ws_cube = (fut_ws ** 3).astype(np.float32)
            fut_sin = sin_wd[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_cos = cos_wd[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_zero = is_zero_power[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_full = is_full_power[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_ramp = is_ramp_zone[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_ws_clip = ws_clipped[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_ws_clip_cube = (fut_ws_clip ** 3).astype(np.float32)

            feat = np.concatenate(
                [
                    np.array(
                        [
                            hist_mean,
                            hist_max,
                            first_diff,
                            ws_curr_cube,
                            sin_curr,
                            cos_curr,
                            zero_curr,
                            full_curr,
                            ramp_curr,
                            ws_clip_curr,
                            ws_clip_curr_cube,
                        ],
                        dtype=np.float32,
                    ),
                    others_ws,
                    fut_ws,
                    fut_ws_cube,
                    fut_sin,
                    fut_cos,
                    fut_zero,
                    fut_full,
                    fut_ramp,
                    fut_ws_clip,
                    fut_ws_clip_cube,
                ],
                axis=0,
            )

            x_rows.append(feat)

    x = np.stack(x_rows, axis=0).astype(np.float32)
    return x


def resolve_model_dir(model_path):
    """使用单一模型路径；若根目录无模型文件则自动选择最新子目录。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型目录: {model_path}")

    scaler_path = os.path.join(model_path, "scaler.pkl")
    model_file_path = os.path.join(model_path, "best_ridge_model.pkl")
    if os.path.exists(scaler_path) and os.path.exists(model_file_path):
        return model_path

    candidates = []
    for name in os.listdir(model_path):
        path = os.path.join(model_path, name)
        if not os.path.isdir(path):
            continue
        scaler_path = os.path.join(path, "scaler.pkl")
        model_file_path = os.path.join(path, "best_ridge_model.pkl")
        if os.path.exists(scaler_path) and os.path.exists(model_file_path):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError("在该模型目录下未找到可用的 scaler.pkl 与 best_ridge_model.pkl")

    candidates.sort()
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="岭回归纯推理脚本")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/wind_test_cleaned.npy",
        help="输入 npy 数据路径",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="saved_models/ridge_regression",
        help="岭回归模型路径（可为直接模型目录或其父目录）",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="预测步长，默认 24",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="saved_models/ridge_regression/ridge_predictions_test_cleaned.npy",
        help="预测结果保存路径",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"找不到数据文件: {args.data_path}")

    model_dir = resolve_model_dir(args.model_path)
    print(f"📦 使用模型目录: {model_dir}")

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    model_path = os.path.join(model_dir, "best_ridge_model.pkl")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    raw_test = np.load(args.data_path, allow_pickle=True)
    print(f"🧪 测试集时间步: {raw_test.shape[0]}")

    x_test = build_ridge_infer_features(raw_test, horizon=args.horizon)
    x_test_scaled = scaler.transform(x_test)

    pred_flat = model.predict(x_test_scaled).astype(np.float32)
    # 方案1：推理后做物理约束裁剪，避免负功率。
    pred_flat = np.clip(pred_flat, 0.0, None)

    num_farms = raw_test.shape[1]
    if num_farms != 10:
        raise ValueError(f"期望风场数为 10，实际为 {num_farms}")

    num_windows = raw_test.shape[0] - args.horizon - 2
    if num_windows <= 0:
        raise ValueError("测试段长度不足以构建推理窗口，请减小 horizon 或增大测试段")

    if pred_flat.ndim != 2 or pred_flat.shape[1] != args.horizon:
        raise ValueError(
            f"模型输出维度异常，期望二维且最后一维为 {args.horizon}，实际为 {pred_flat.shape}"
        )

    expected_rows = num_windows * num_farms
    if pred_flat.shape[0] != expected_rows:
        raise ValueError(
            f"模型输出行数异常，期望 {expected_rows}，实际 {pred_flat.shape[0]}"
        )

    predictions = pred_flat.reshape(num_windows, 10, args.horizon, order="C")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(args.output, predictions)

    print("\n" + "=" * 40)
    print(f"🎉 预测完成，结果已保存: {args.output}")
    print(f"📐 输出形状: {predictions.shape} (Sample, Farm, Horizon)")
    print("=" * 40)


if __name__ == "__main__":
    main()
