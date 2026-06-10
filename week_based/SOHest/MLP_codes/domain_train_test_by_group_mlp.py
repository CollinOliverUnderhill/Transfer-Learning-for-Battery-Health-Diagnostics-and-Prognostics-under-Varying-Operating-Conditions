#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RIDGE_UTIL_PATH = Path(__file__).resolve().parents[1] / "Ridge_codes" / "domain_train_test_by_group.py"
if not RIDGE_UTIL_PATH.exists():
    raise FileNotFoundError(f"Required ridge utility script not found: {RIDGE_UTIL_PATH}")

_RIDGE_SPEC = importlib.util.spec_from_file_location("ridge_domain_utils", RIDGE_UTIL_PATH)
if _RIDGE_SPEC is None or _RIDGE_SPEC.loader is None:
    raise ImportError(f"Failed to load ridge utility module from: {RIDGE_UTIL_PATH}")
ridge_utils = importlib.util.module_from_spec(_RIDGE_SPEC)
_RIDGE_SPEC.loader.exec_module(ridge_utils)


def parse_hidden_dims(arg: str) -> List[int]:
    vals = [x.strip() for x in str(arg).split(",") if x.strip()]
    if not vals:
        raise ValueError("--hidden_dims must contain at least one integer, e.g. 64,32")
    dims = [int(v) for v in vals]
    if any(d <= 0 for d in dims):
        raise ValueError("--hidden_dims must be positive integers.")
    return dims


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    s = str(device_arg).strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(s)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in the current environment.")
    return device


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float, activation: str) -> None:
        super().__init__()
        act_name = str(activation).strip().lower()
        if act_name == "relu":
            act_cls = nn.ReLU
        elif act_name == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(act_cls())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def freeze_first_hidden_layers(model: MLPRegressor, n_hidden_layers_to_freeze: int) -> List[str]:
    if n_hidden_layers_to_freeze <= 0:
        return []
    frozen_names: List[str] = []
    linear_idx = 0
    for module in model.net:
        if isinstance(module, nn.Linear):
            is_output = linear_idx == sum(isinstance(m, nn.Linear) for m in model.net) - 1
            if is_output:
                break
            if linear_idx < int(n_hidden_layers_to_freeze):
                for name, param in module.named_parameters(recurse=False):
                    param.requires_grad = False
                    frozen_names.append(f"hidden_linear_{linear_idx}.{name}")
            linear_idx += 1
    return frozen_names


def split_train_val_by_cell(df: pd.DataFrame, val_cell_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < float(val_cell_frac) < 1.0):
        return df.copy(), df.iloc[0:0].copy()
    cells = sorted(df["cell"].unique().tolist(), key=ridge_utils.cell_sort_key)
    if len(cells) < 2:
        return df.copy(), df.iloc[0:0].copy()

    rng = np.random.default_rng(int(seed))
    cells_arr = np.array(cells, dtype=object)
    rng.shuffle(cells_arr)
    n_val = int(round(len(cells_arr) * float(val_cell_frac)))
    n_val = max(1, min(n_val, len(cells_arr) - 1))
    val_cells = set(cells_arr[:n_val].tolist())
    train_df = df[~df["cell"].isin(val_cells)].copy()
    val_df = df[df["cell"].isin(val_cells)].copy()
    if len(train_df) == 0 or len(val_df) == 0:
        return df.copy(), df.iloc[0:0].copy()
    return train_df, val_df


def df_to_tensors(
    df: pd.DataFrame,
    x_cols: Sequence[str],
    y_col: str,
    mu: np.ndarray,
    sd: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = df[list(x_cols)].to_numpy(dtype=float)
    X_s = ridge_utils.standardize_apply(X, mu, sd)
    y = df[y_col].to_numpy(dtype=float)
    return (
        torch.tensor(X_s, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
    )


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def predict_with_model(
    model: nn.Module,
    df: pd.DataFrame,
    x_cols: Sequence[str],
    mu: np.ndarray,
    sd: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if len(df) == 0:
        return np.empty((0,), dtype=float)

    model.eval()
    X = df[list(x_cols)].to_numpy(dtype=float)
    X_s = ridge_utils.standardize_apply(X, mu, sd)
    X_t = torch.tensor(X_s, dtype=torch.float32, device=device)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_t), max(1, int(batch_size))):
            batch = X_t[start : start + max(1, int(batch_size))]
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_loss(
    model: nn.Module,
    df: pd.DataFrame,
    x_cols: Sequence[str],
    y_col: str,
    mu: np.ndarray,
    sd: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    if len(df) == 0:
        return float("nan")
    model.eval()
    X_t, y_t = df_to_tensors(df, x_cols, y_col, mu, sd, device)
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for start in range(0, len(X_t), max(1, int(batch_size))):
            xb = X_t[start : start + max(1, int(batch_size))]
            yb = y_t[start : start + max(1, int(batch_size))]
            total_loss += float(loss_fn(model(xb), yb).item()) * len(xb)
            total_n += len(xb)
    return total_loss / total_n if total_n > 0 else float("nan")


def train_stage(
    *,
    train_df: pd.DataFrame,
    x_cols: Sequence[str],
    y_col: str,
    mu: np.ndarray,
    sd: np.ndarray,
    hidden_dims: Sequence[int],
    dropout: float,
    activation: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    val_cell_frac: float,
    early_stop_patience: int,
    min_epochs_before_early_stop: int,
    seed: int,
    device: torch.device,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    freeze_hidden_layers: int = 0,
    log_every: int = 1,
    stage_name: str = "train",
    require_validation: bool = False,
) -> Dict[str, object]:
    set_seed(int(seed))
    train_part_df, val_part_df = split_train_val_by_cell(train_df, val_cell_frac=val_cell_frac, seed=seed)
    if require_validation and len(val_part_df) == 0:
        raise ValueError(
            f"stage={stage_name} requires a non-empty validation split, "
            f"but val_cell_frac={float(val_cell_frac)} produced no validation cells."
        )
    model = MLPRegressor(len(x_cols), hidden_dims, dropout=dropout, activation=activation).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    frozen_names = freeze_first_hidden_layers(model, int(freeze_hidden_layers))

    X_train_t, y_train_t = df_to_tensors(train_part_df, x_cols, y_col, mu, sd, device)
    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=max(1, min(int(batch_size), len(train_part_df))),
        shuffle=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after freezing layers.")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    use_val = len(val_part_df) > 0
    best_metric = float("inf")
    best_epoch = 0
    best_state = clone_state_dict(model)
    history: List[Dict[str, float]] = []
    patience_left = int(early_stop_patience)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)

        train_loss = running_loss / n_seen if n_seen > 0 else float("nan")
        val_loss = evaluate_loss(model, val_part_df, x_cols, y_col, mu, sd, device, batch_size) if use_val else train_loss
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})
        if int(log_every) > 0 and (epoch == 1 or epoch % int(log_every) == 0):
            print(
                f"[LOSS] stage={stage_name} epoch={epoch} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

        if val_loss < best_metric:
            best_metric = float(val_loss)
            best_epoch = int(epoch)
            best_state = clone_state_dict(model)
            patience_left = int(early_stop_patience)
        else:
            if epoch >= int(min_epochs_before_early_stop):
                patience_left -= 1
                if patience_left <= 0:
                    break

    model.load_state_dict(best_state)
    y_pred_train = predict_with_model(model, train_df, x_cols, mu, sd, device, batch_size)
    return {
        "model": model,
        "state_dict": best_state,
        "y_pred_train": y_pred_train,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "used_validation": bool(use_val),
        "freeze_hidden_layers": int(freeze_hidden_layers),
        "frozen_parameter_names": frozen_names,
        "train_cells": sorted(train_part_df["cell"].astype(str).unique().tolist(), key=ridge_utils.cell_sort_key),
        "val_cells": sorted(val_part_df["cell"].astype(str).unique().tolist(), key=ridge_utils.cell_sort_key),
    }


def save_torch_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def maybe_plot_training_history(history: List[Dict[str, float]], out_path: Path, title: str) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[WARN] matplotlib is not installed. Skip training history plot generation.")
        return

    hist_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(hist_df["epoch"], hist_df["train_loss"], color="#F58518", linewidth=1.8, label="Train loss")
    ax.plot(hist_df["epoch"], hist_df["val_loss"], color="#4C78A8", linewidth=1.8, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MLP domain train/test workflow for IVAS battery groups and conditions.")
    ap.add_argument("--data_csv", type=str, default="E:/Datasets/IVAS/Processing_Data/rpt_samples_feature_soh.csv")
    ap.add_argument("--group_cond_csv", type=str, default="E:/Datasets/IVAS/Groupcondi.csv")
    ap.add_argument("--out_dir", type=str, default="E:/Datasets/IVAS/MLP_Results_DomainShift")
    ap.add_argument("--split_mode", type=str, default="explicit", choices=["explicit", "within_group_random"])
    ap.add_argument("--train_groups", type=str, default="")
    ap.add_argument("--test_groups", type=str, default="")
    ap.add_argument("--train_cells", type=str, default="")
    ap.add_argument("--test_cells", type=str, default="")
    ap.add_argument(
        "--target_ft_groups",
        type=str,
        default="",
        help='Optional target-domain fine-tune groups, chosen from eligible groups outside both train and explicit test pools.',
    )
    ap.add_argument("--target_ft_group_count", type=int, default=0)
    ap.add_argument("--target_ft_seed", type=int, default=42)
    ap.add_argument("--groups", type=str, default="all")
    ap.add_argument("--test_cell_frac", type=float, default=0.5)
    ap.add_argument("--cell_split_seed", type=int, default=42)
    ap.add_argument("--train_release", type=str, default="all")
    ap.add_argument("--test_release", type=str, default="all")
    ap.add_argument("--x_cols", type=str, default="feature_mean_ic")
    ap.add_argument("--y_col", type=str, default="soh")
    ap.add_argument("--sort_col", type=str, default="time_week")
    ap.add_argument("--time_col", type=str, default="time_week")
    ap.add_argument("--hidden_dims", type=str, default="128,96,64,32")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--ft_epochs", type=int, default=60)
    ap.add_argument("--ft_freeze_hidden_layers", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ft_weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_cell_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=25)
    ap.add_argument("--min_epochs_before_early_stop", type=int, default=0)
    ap.add_argument("--ft_min_epochs_before_early_stop", type=int, default=-1)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--tail_q", type=float, default=0.95)
    ap.add_argument("--min_train_rows", type=int, default=8)
    ap.add_argument("--min_test_rows", type=int, default=8)
    ap.add_argument("--save_predictions", action="store_true")
    ap.add_argument("--plot", dest="plot", action="store_true", default=True)
    ap.add_argument("--no_plot", dest="plot", action="store_false")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if not (0.0 < float(args.tail_q) < 1.0):
        raise ValueError("--tail_q must be in (0,1).")
    if int(args.target_ft_group_count) < 0:
        raise ValueError("--target_ft_group_count must be non-negative.")
    if int(args.ft_freeze_hidden_layers) < 0:
        raise ValueError("--ft_freeze_hidden_layers must be non-negative.")
    if int(args.ft_min_epochs_before_early_stop) < -1:
        raise ValueError("--ft_min_epochs_before_early_stop must be -1 or a non-negative integer.")

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    data_csv = Path(args.data_csv)
    group_cond_csv = Path(args.group_cond_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw, x_cols = ridge_utils.read_samples_csv(data_csv, args.x_cols, args.y_col, args.sort_col, args.time_col)
    cond_df = ridge_utils.load_group_conditions(group_cond_csv)
    train_release_filter = ridge_utils.parse_release_filter(args.train_release)
    test_release_filter = ridge_utils.parse_release_filter(args.test_release)
    train_df_all = ridge_utils.add_condition_columns(ridge_utils.apply_release_filter(df_raw, train_release_filter), cond_df)
    test_df_all = ridge_utils.add_condition_columns(ridge_utils.apply_release_filter(df_raw, test_release_filter), cond_df)
    train_cell_df = ridge_utils.filter_cells_by_min_rows(ridge_utils.build_cell_table(train_df_all), args.min_train_rows)
    test_cell_df = ridge_utils.filter_cells_by_min_rows(ridge_utils.build_cell_table(test_df_all), args.min_test_rows)

    if args.split_mode == "explicit":
        train_cells, candidate_test_cells, split_info = ridge_utils.resolve_explicit_split(train_cell_df, test_cell_df, args)
    else:
        shared_cells = sorted(
            set(train_cell_df["cell"].unique().tolist()) & set(test_cell_df["cell"].unique().tolist()),
            key=ridge_utils.cell_sort_key,
        )
        shared_df = train_cell_df[train_cell_df["cell"].isin(shared_cells)].copy()
        if len(shared_df) == 0:
            raise ValueError("within_group_random found no shared eligible cells between train/test release filters.")
        train_cells, candidate_test_cells, split_info = ridge_utils.resolve_within_group_random_split(shared_df, args)

    target_ft_cells, target_test_cells, target_ft_info = ridge_utils.resolve_target_finetune_split(
        test_cell_df=test_cell_df,
        train_cells=train_cells,
        candidate_test_cells=candidate_test_cells,
        args=args,
    )
    split_info["target_ft"] = target_ft_info
    source_train_df = ridge_utils.materialize_cells_frame(train_df_all, train_cells, int(args.min_train_rows), "Source train")
    target_ft_df = ridge_utils.materialize_cells_frame(test_df_all, target_ft_cells, int(args.min_train_rows), "Target fine-tune", allow_empty=True)
    target_test_df = ridge_utils.materialize_cells_frame(test_df_all, target_test_cells, int(args.min_test_rows), "Target holdout test")

    mu, sd = ridge_utils.standardize_fit(source_train_df[list(x_cols)].to_numpy(dtype=float))
    source_stage = train_stage(
        train_df=source_train_df,
        x_cols=x_cols,
        y_col=args.y_col,
        mu=mu,
        sd=sd,
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
        activation=args.activation,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        val_cell_frac=float(args.val_cell_frac),
        early_stop_patience=int(args.early_stop_patience),
        min_epochs_before_early_stop=int(args.min_epochs_before_early_stop),
        seed=int(args.seed),
        device=device,
        freeze_hidden_layers=0,
        log_every=int(args.log_every),
        stage_name="source_train",
    )
    source_train_pred_df = ridge_utils.build_prediction_df(source_train_df, args.y_col, np.asarray(source_stage["y_pred_train"], dtype=float), "source_train")
    target_test_pred_df_source = ridge_utils.build_prediction_df(
        target_test_df,
        args.y_col,
        predict_with_model(source_stage["model"], target_test_df, x_cols, mu, sd, device, int(args.batch_size)),
        "target_test_source_only",
    )

    has_target_finetune = len(target_ft_df) > 0
    ft_min_epochs_before_early_stop = (
        int(args.min_epochs_before_early_stop)
        if int(args.ft_min_epochs_before_early_stop) < 0
        else int(args.ft_min_epochs_before_early_stop)
    )
    active_stage = source_stage
    target_ft_pred_df = pd.DataFrame()
    target_test_pred_df_final = target_test_pred_df_source.copy()
    if has_target_finetune:
        active_stage = train_stage(
            train_df=target_ft_df,
            x_cols=x_cols,
            y_col=args.y_col,
            mu=mu,
            sd=sd,
            hidden_dims=hidden_dims,
            dropout=float(args.dropout),
            activation=args.activation,
            epochs=int(args.ft_epochs),
            lr=float(args.ft_lr),
            weight_decay=float(args.ft_weight_decay),
            batch_size=int(args.batch_size),
            val_cell_frac=float(args.val_cell_frac),
            early_stop_patience=int(args.early_stop_patience),
            min_epochs_before_early_stop=ft_min_epochs_before_early_stop,
            seed=int(args.seed) + 1,
            device=device,
            init_state_dict=source_stage["state_dict"],
            freeze_hidden_layers=int(args.ft_freeze_hidden_layers),
            log_every=int(args.log_every),
            stage_name="target_finetune",
        )
        target_ft_pred_df = ridge_utils.build_prediction_df(target_ft_df, args.y_col, np.asarray(active_stage["y_pred_train"], dtype=float), "target_finetune")
        target_test_pred_df_final = ridge_utils.build_prediction_df(
            target_test_df,
            args.y_col,
            predict_with_model(active_stage["model"], target_test_df, x_cols, mu, sd, device, int(args.batch_size)),
            "target_test_finetuned",
        )

    source_train_overall_df = ridge_utils.summarize_overall(source_train_pred_df, "source_train", float(args.tail_q))
    target_test_source_overall_df = ridge_utils.summarize_overall(target_test_pred_df_source, "target_test_source_only", float(args.tail_q))
    target_test_overall_df = ridge_utils.summarize_overall(target_test_pred_df_final, "target_test_finetuned" if has_target_finetune else "test", float(args.tail_q))
    source_train_cell_metrics_df = ridge_utils.summarize_by_cell(source_train_pred_df, args.sort_col, args.time_col, float(args.tail_q))
    target_test_source_cell_metrics_df = ridge_utils.summarize_by_cell(target_test_pred_df_source, args.sort_col, args.time_col, float(args.tail_q))
    target_test_cell_metrics_df = ridge_utils.summarize_by_cell(target_test_pred_df_final, args.sort_col, args.time_col, float(args.tail_q))
    source_train_group_metrics_df = ridge_utils.summarize_by_group(source_train_pred_df, source_train_cell_metrics_df, float(args.tail_q))
    target_test_source_group_metrics_df = ridge_utils.summarize_by_group(target_test_pred_df_source, target_test_source_cell_metrics_df, float(args.tail_q))
    target_test_group_metrics_df = ridge_utils.summarize_by_group(target_test_pred_df_final, target_test_cell_metrics_df, float(args.tail_q))
    target_ft_overall_df = ridge_utils.summarize_overall(target_ft_pred_df, "target_finetune", float(args.tail_q)) if has_target_finetune else pd.DataFrame()
    target_ft_cell_metrics_df = ridge_utils.summarize_by_cell(target_ft_pred_df, args.sort_col, args.time_col, float(args.tail_q)) if has_target_finetune else pd.DataFrame()
    target_ft_group_metrics_df = ridge_utils.summarize_by_group(target_ft_pred_df, target_ft_cell_metrics_df, float(args.tail_q)) if has_target_finetune else pd.DataFrame()

    split_train_cells_df = ridge_utils.build_cell_table(source_train_df).sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)
    split_test_cells_df = ridge_utils.build_cell_table(target_test_df).sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)
    split_target_ft_cells_df = ridge_utils.build_cell_table(target_ft_df).sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True) if has_target_finetune else pd.DataFrame(columns=split_train_cells_df.columns)

    meta = {
        "model_type": "mlp_regressor",
        "x_cols": list(x_cols),
        "y_col": str(args.y_col),
        "hidden_dims": list(hidden_dims),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "mu": np.asarray(mu, dtype=float).tolist(),
        "sd": np.asarray(sd, dtype=float).tolist(),
        "split_info": split_info,
        "source_history": source_stage["history"],
        "active_history": active_stage["history"],
        "source_freeze_hidden_layers": int(source_stage["freeze_hidden_layers"]),
        "active_freeze_hidden_layers": int(active_stage["freeze_hidden_layers"]),
        "active_frozen_parameter_names": list(active_stage["frozen_parameter_names"]),
    }

    split_train_cells_df.to_csv(out_dir / "train_cells.csv", index=False)
    split_test_cells_df.to_csv(out_dir / "test_cells.csv", index=False)
    pd.DataFrame(source_stage["history"]).to_csv(out_dir / "training_history_source.csv", index=False)
    maybe_plot_training_history(
        source_stage["history"],
        out_dir / "plot_training_history_source.png",
        "Source Train: Loss vs Epoch",
    )
    source_train_overall_df.to_csv(out_dir / "train_overall_metrics.csv", index=False)
    target_test_overall_df.to_csv(out_dir / "test_overall_metrics.csv", index=False)
    source_train_cell_metrics_df.to_csv(out_dir / "train_cell_metrics.csv", index=False)
    target_test_cell_metrics_df.to_csv(out_dir / "test_cell_metrics.csv", index=False)
    source_train_group_metrics_df.to_csv(out_dir / "train_group_metrics.csv", index=False)
    target_test_group_metrics_df.to_csv(out_dir / "test_group_metrics.csv", index=False)
    if has_target_finetune:
        pd.DataFrame(active_stage["history"]).to_csv(out_dir / "training_history_finetune.csv", index=False)
        maybe_plot_training_history(
            active_stage["history"],
            out_dir / "plot_training_history_finetune.png",
            "Target Fine-tune: Loss vs Epoch",
        )
        target_test_source_overall_df.to_csv(out_dir / "test_overall_metrics_source_only.csv", index=False)
        target_test_source_cell_metrics_df.to_csv(out_dir / "test_cell_metrics_source_only.csv", index=False)
        target_test_source_group_metrics_df.to_csv(out_dir / "test_group_metrics_source_only.csv", index=False)
        split_target_ft_cells_df.to_csv(out_dir / "target_finetune_cells.csv", index=False)
        target_ft_overall_df.to_csv(out_dir / "target_finetune_overall_metrics.csv", index=False)
        target_ft_cell_metrics_df.to_csv(out_dir / "target_finetune_cell_metrics.csv", index=False)
        target_ft_group_metrics_df.to_csv(out_dir / "target_finetune_group_metrics.csv", index=False)
    if args.save_predictions:
        source_train_pred_df.to_csv(out_dir / "predictions_train.csv", index=False)
        target_test_pred_df_final.to_csv(out_dir / "predictions_test.csv", index=False)
        if has_target_finetune:
            target_test_pred_df_source.to_csv(out_dir / "predictions_test_source_only.csv", index=False)
            target_ft_pred_df.to_csv(out_dir / "predictions_target_finetune.csv", index=False)

    save_torch_checkpoint(out_dir / "model.pt", {"meta": meta, "state_dict": active_stage["state_dict"]})
    ridge_utils.save_json(out_dir / "model.json", meta)
    ridge_utils.save_json(out_dir / "config.json", {"data_csv": str(data_csv), "group_cond_csv": str(group_cond_csv), "seed": int(args.seed), "device": str(device), "split_info": split_info})
    if has_target_finetune:
        save_torch_checkpoint(out_dir / "model_source.pt", {"meta": meta, "state_dict": source_stage["state_dict"]})
        save_torch_checkpoint(out_dir / "model_finetuned.pt", {"meta": meta, "state_dict": active_stage["state_dict"]})

    if args.plot:
        ridge_utils.maybe_make_plots(source_train_pred_df, target_ft_pred_df, target_test_pred_df_final, target_test_cell_metrics_df, target_test_group_metrics_df, target_test_overall_df, out_dir)

    print("[INFO] MLP domain train/test finished.")
    print(f"[INFO] device            : {device}")
    print(f"[INFO] x_cols            : {x_cols}")
    print(f"[INFO] hidden_dims       : {hidden_dims}")
    print(f"[INFO] ft freeze layers  : {int(args.ft_freeze_hidden_layers)}")
    print(f"[INFO] saved out_dir     : {out_dir}")
    print(f"[INFO] train overall     : {ridge_utils.format_overall_metrics_line(source_train_overall_df.iloc[0])}")
    if has_target_finetune:
        print(f"[INFO] target ft overall : {ridge_utils.format_overall_metrics_line(target_ft_overall_df.iloc[0])}")
        print(f"[INFO] test source-only  : {ridge_utils.format_overall_metrics_line(target_test_source_overall_df.iloc[0])}")
        print(f"[INFO] test overall      : {ridge_utils.format_overall_metrics_line(target_test_overall_df.iloc[0])}")
    else:
        print(f"[INFO] test overall      : {ridge_utils.format_overall_metrics_line(target_test_overall_df.iloc[0])}")


if __name__ == "__main__":
    main()
