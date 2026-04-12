import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


def process_advanced_features(raw_data):
    """
    输入:
        raw_data: numpy array, 形状 [T, N, F_raw], 其中至少包含:
          - index 0: 时间戳(可被 pandas 解析)
          - index 1: U10
          - index 2: V10
          - index 3: U100
          - index 4: V100
          - index 5: Power

    输出:
        features: numpy array, 形状 [T, N, 13]
        特征顺序:
          [U10, V10, U100, V100, WS10, WD10, WS100, WD100,
           Power, hour_sin, hour_cos, month_sin, month_cos]
    """
    if raw_data.ndim != 3:
        raise ValueError(f"raw_data 应为 3 维 [T, N, F_raw]，实际是 {raw_data.shape}")
    if raw_data.shape[-1] < 6:
        raise ValueError("raw_data 最后一维至少需要包含 0..5 号字段")

    u10 = raw_data[:, :, 1].astype(np.float32)
    v10 = raw_data[:, :, 2].astype(np.float32)
    u100 = raw_data[:, :, 3].astype(np.float32)
    v100 = raw_data[:, :, 4].astype(np.float32)
    power = raw_data[:, :, 5].astype(np.float32)

    ws10 = np.sqrt(u10 ** 2 + v10 ** 2).astype(np.float32)
    wd10 = np.arctan2(v10, u10).astype(np.float32)
    ws100 = np.sqrt(u100 ** 2 + v100 ** 2).astype(np.float32)
    wd100 = np.arctan2(v100, u100).astype(np.float32)

    # 只取任意节点的时间列（通常各节点时间同步），按 [T] 构建时间周期特征
    time_col = raw_data[:, 0, 0]
    dt = pd.to_datetime(time_col)

    hour = dt.hour.to_numpy(dtype=np.float32)
    month = dt.month.to_numpy(dtype=np.float32)

    hour_sin = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    month_sin = np.sin(2.0 * np.pi * (month - 1.0) / 12.0).astype(np.float32)
    month_cos = np.cos(2.0 * np.pi * (month - 1.0) / 12.0).astype(np.float32)

    # 扩展到 [T, N]
    num_nodes = raw_data.shape[1]
    hour_sin = np.repeat(hour_sin[:, None], num_nodes, axis=1)
    hour_cos = np.repeat(hour_cos[:, None], num_nodes, axis=1)
    month_sin = np.repeat(month_sin[:, None], num_nodes, axis=1)
    month_cos = np.repeat(month_cos[:, None], num_nodes, axis=1)

    features = np.stack(
        [
            u10,
            v10,
            u100,
            v100,
            ws10,
            wd10,
            ws100,
            wd100,
            power,
            hour_sin,
            hour_cos,
            month_sin,
            month_cos,
        ],
        axis=-1,
    ).astype(np.float32)

    return features


class GraphLearning(nn.Module):
    """
    自适应图学习: A = relu(tanh(alpha * E1 @ E2^T))
    输出 A 形状 [N, N]
    """

    def __init__(self, num_nodes, embed_dim=10, alpha=3.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.alpha = alpha

        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))

    def forward(self):
        logits = self.alpha * torch.mm(self.E1, self.E2.t())
        A = F.relu(torch.tanh(logits))
        return A


class TemporalInception(nn.Module):
    """
    输入 x: [B, C, N, T]
    输出 y: [B, c_out, N, T]
    """

    def __init__(self, c_in, c_out):
        super().__init__()

        base = c_out // 4
        rem = c_out % 4
        branch_channels = [base + (1 if i < rem else 0) for i in range(4)]

        self.conv_k2 = nn.Conv2d(c_in, branch_channels[0], kernel_size=(1, 2), padding="same")
        self.conv_k3 = nn.Conv2d(c_in, branch_channels[1], kernel_size=(1, 3), padding="same")
        self.conv_k5 = nn.Conv2d(c_in, branch_channels[2], kernel_size=(1, 5), padding="same")
        self.conv_k7 = nn.Conv2d(c_in, branch_channels[3], kernel_size=(1, 7), padding="same")

    def forward(self, x):
        x2 = self.conv_k2(x)
        x3 = self.conv_k3(x)
        x5 = self.conv_k5(x)
        x7 = self.conv_k7(x)
        out = torch.cat([x2, x3, x5, x7], dim=1)
        return out


class SpatialGCN(nn.Module):
    """
    输入 x: [B, C, N, T]
    输入 A: [N, N]
    输出 y: [B, c_out, N, T]
    """

    def __init__(self, c_in, c_out):
        super().__init__()
        self.theta = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x, A):
        support = self.theta(x)  # [B, C_out, N, T]

        # 按题目给定 einsum 形式聚合邻域: 先转到 [B, C, T, N]
        support_tn = support.permute(0, 1, 3, 2)
        out_tn = torch.einsum("vw,bcfw->bcfv", A, support_tn)
        out = out_tn.permute(0, 1, 3, 2)

        return self.act(out)


class STGCNBlock(nn.Module):
    """TemporalInception -> SpatialGCN -> TemporalInception"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.t1 = TemporalInception(hidden_dim, hidden_dim)
        self.s = SpatialGCN(hidden_dim, hidden_dim)
        self.t2 = TemporalInception(hidden_dim, hidden_dim)
        self.norm = nn.BatchNorm2d(hidden_dim)

    def forward(self, x, A):
        out = self.t1(x)
        out = self.s(out, A)
        out = self.t2(out)
        out = self.norm(out)
        return out


class WindSTGCN(nn.Module):
    """
    输入 x: [B, Seq_Len(48), Nodes(10), Features(13)]
    输出 y: [B, Nodes, output_dim(24)]
    """

    def __init__(self, num_nodes=10, in_dim=13, hidden_dim=64, output_dim=24):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        self.graph_learning = GraphLearning(num_nodes=num_nodes, embed_dim=10, alpha=3.0)

        self.block = STGCNBlock(hidden_dim=hidden_dim)

        self.dropout = nn.Dropout(p=0.1)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"x 必须是 4 维 [B, T, N, F]，实际是 {x.shape}")

        # [B, T, N, F] -> [B, F, N, T]
        x = x.permute(0, 3, 2, 1)

        x = self.input_proj(x)

        A = self.graph_learning()

        res = x
        x = self.block(x, A)
        x = x + res
        x = self.dropout(x)

        # 取最后一个时间步做解码: [B, H, N, T] -> [B, N, H]
        last_step = x[:, :, :, -1].permute(0, 2, 1)

        # [B, N, H] -> [B, N, output_dim]
        out = self.decoder(last_step)
        return out


class WindSTGCNBaseline(WindSTGCN):
    """版本命名别名，便于实验管理。"""


if __name__ == "__main__":
    # 简单形状自检
    model = WindSTGCN(num_nodes=10, in_dim=13, hidden_dim=64, output_dim=24)
    x = torch.randn(8, 48, 10, 13)
    y = model(x)
    print("Output shape:", tuple(y.shape))  # (8, 10, 24)
