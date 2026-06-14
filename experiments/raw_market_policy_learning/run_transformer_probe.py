from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TinyWindowTransformer(nn.Module):
    def __init__(
        self,
        feature_count: int,
        horizon_count: int,
        action_count: int,
        lookback: int,
        d_model: int = 32,
        nhead: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(feature_count, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon_count * action_count),
        )
        self.horizon_count = horizon_count
        self.action_count = action_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input(x) + self.pos[:, : x.shape[1], :]
        z = self.encoder(z)
        pooled = z[:, -1, :]
        y = self.head(pooled)
        return y.reshape(-1, self.horizon_count, self.action_count)


def load_examples(path: str, train_frac: float):
    data = np.load(path)
    x = data["x"].astype(np.float32)  # samples, instruments, lookback, features
    reward = data["reward"].astype(np.float32)  # samples, instruments, horizons, actions
    sample_times = data["sample_times"]
    instruments = data["instruments"]
    horizons = data["horizons"]
    actions = data["actions"]

    split = int(x.shape[0] * train_frac)
    train_x = x[:split].reshape(-1, x.shape[2], x.shape[3])
    train_y = reward[:split].reshape(-1, reward.shape[2], reward.shape[3])
    test_x = x[split:].reshape(-1, x.shape[2], x.shape[3])
    test_y = reward[split:].reshape(-1, reward.shape[2], reward.shape[3])

    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std[std == 0] = 1.0
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "sample_times": sample_times,
        "instruments": instruments,
        "horizons": horizons,
        "actions": actions,
        "split_sample": split,
        "split_time": sample_times[split],
        "raw_shape": x.shape,
        "reward_shape": reward.shape,
    }


def policy_rewards(pred: np.ndarray, reward: np.ndarray):
    chosen = pred.argmax(axis=2)  # examples, horizons
    out = {}
    for h in range(reward.shape[1]):
        idx = chosen[:, h]
        selected = reward[np.arange(len(reward)), h, idx]
        out[h] = {
            "transformer_argmax_reward": selected,
            "chosen_actions": idx,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    loaded = load_examples(args.input, args.train_frac)
    train_x = torch.from_numpy(loaded["train_x"])
    train_y = torch.from_numpy(loaded["train_y"])
    test_x = torch.from_numpy(loaded["test_x"])
    test_y = loaded["test_y"]
    horizons = loaded["horizons"]
    actions = loaded["actions"]

    model = TinyWindowTransformer(
        feature_count=train_x.shape[-1],
        horizon_count=train_y.shape[1],
        action_count=train_y.shape[2],
        lookback=train_x.shape[1],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    losses = []
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        losses.append(total / count)
        print(f"epoch={epoch + 1} train_mse={losses[-1]:.8f}", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_x), args.batch_size):
            preds.append(model(test_x[start : start + args.batch_size]).numpy())
    pred = np.concatenate(preds, axis=0)

    policy = policy_rewards(pred, test_y)
    lines = []
    mix_lines = []
    for h_idx, horizon in enumerate(horizons):
        baseline = {
            "always_long": test_y[:, h_idx, 0],
            "always_short": test_y[:, h_idx, 1],
            "always_flat": test_y[:, h_idx, 2],
            "transformer_argmax_reward": policy[h_idx]["transformer_argmax_reward"],
        }
        for name, values in baseline.items():
            lines.append(
                f"{int(horizon)},{name},{len(values)},{values.mean():.8f},{values.std():.8f}"
            )
        chosen = policy[h_idx]["chosen_actions"]
        for action_idx, action in enumerate(actions):
            mask = chosen == action_idx
            mean = policy[h_idx]["transformer_argmax_reward"][mask].mean() if mask.any() else 0.0
            mix_lines.append(f"{int(horizon)},{action},{int(mask.sum())},{mean:.8f}")

    out = Path(args.summary)
    out.write_text(
        "\n".join(
            [
                "# Transformer Probe",
                "",
                f"- input: {args.input}",
                f"- raw_shape: {loaded['raw_shape']}",
                f"- reward_shape: {loaded['reward_shape']}",
                f"- train_fraction: {args.train_frac}",
                f"- split_time: {loaded['split_time']}",
                f"- epochs: {args.epochs}",
                f"- batch_size: {args.batch_size}",
                "- model: tiny per-instrument Transformer encoder over raw lookback windows",
                "- tuning: none",
                "",
                "## Train Loss",
                "",
                "```csv",
                "epoch,train_mse",
                "\n".join(f"{i + 1},{v:.8f}" for i, v in enumerate(losses)),
                "```",
                "",
                "## Test Reward By Policy",
                "",
                "```csv",
                "horizon,policy,count,mean_reward,std_reward",
                "\n".join(lines),
                "```",
                "",
                "## Transformer Action Mix",
                "",
                "```csv",
                "horizon,action,count,mean_reward",
                "\n".join(mix_lines),
                "```",
                "",
                "## Guard",
                "",
                "This is a fixed Transformer probe. It is not a strategy and does not tune",
                "features, horizons, architecture, or thresholds.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
