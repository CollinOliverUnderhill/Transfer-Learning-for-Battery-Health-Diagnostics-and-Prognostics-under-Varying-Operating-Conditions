#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from three_step_common import mlp_base, save_json, select_support_cells, week_runner


def average_state_dicts(states: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("Cannot average an empty state list.")
    averaged: Dict[str, torch.Tensor] = {}
    for key in states[0]:
        first = states[0][key]
        if torch.is_floating_point(first):
            stacked = torch.stack([state[key].to(dtype=torch.float32) for state in states], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=first.dtype)
        else:
            averaged[key] = states[-1][key].clone()
    return averaged


def rolling_median(values: Sequence[float], window: int) -> float:
    if not values:
        return float("inf")
    n = max(1, int(window))
    return float(np.median(np.asarray(values[-n:], dtype=float)))


def train_stage_with_replay(
    *,
    train_df: pd.DataFrame,
    replay_df: Optional[pd.DataFrame],
    replay_loss_weight: float,
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
    init_state_dict: Dict[str, torch.Tensor],
    freeze_hidden_layers: int,
    min_val_cells: int,
    log_every: int,
    stage_name: str,
    ft_batch_mode: str = "mini",
    ft_selection_mode: str = "raw_best",
    ft_smooth_window: int = 25,
    ft_swa_window: int = 50,
    ft_l2sp_weight: float = 0.0,
) -> Dict[str, object]:
    mlp_base.set_seed(int(seed))
    train_part_df, val_part_df = mlp_base.split_train_val_by_cell(train_df, val_cell_frac=val_cell_frac, seed=seed)
    if len(val_part_df) == 0:
        raise ValueError(f"stage={stage_name} requires a non-empty validation split.")
    val_cell_count = int(val_part_df["cell"].astype(str).nunique())
    if val_cell_count < int(min_val_cells):
        raise ValueError(
            f"stage={stage_name} target_finetune val cell count {val_cell_count} is below required minimum {int(min_val_cells)}."
        )

    model = mlp_base.MLPRegressor(len(x_cols), hidden_dims, dropout=dropout, activation=activation).to(device)
    model.load_state_dict(init_state_dict)
    frozen_names = mlp_base.freeze_first_hidden_layers(model, int(freeze_hidden_layers))

    X_train_t, y_train_t = mlp_base.df_to_tensors(train_part_df, x_cols, y_col, mu, sd, device)
    full_batch = str(ft_batch_mode).strip().lower() == "full"
    train_batch_size = len(train_part_df) if full_batch else int(batch_size)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=max(1, min(train_batch_size, len(train_part_df))),
        shuffle=not full_batch,
    )

    use_replay = replay_df is not None and len(replay_df) > 0 and float(replay_loss_weight) > 0.0
    replay_loader = None
    if use_replay:
        X_replay_t, y_replay_t = mlp_base.df_to_tensors(replay_df, x_cols, y_col, mu, sd, device)
        replay_batch_size = len(replay_df) if full_batch else int(batch_size)
        replay_loader = DataLoader(
            TensorDataset(X_replay_t, y_replay_t),
            batch_size=max(1, min(replay_batch_size, len(replay_df))),
            shuffle=not full_batch,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after freezing layers.")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay), foreach=False)
    loss_fn = nn.MSELoss()
    source_params = {
        name: init_state_dict[name].detach().to(device=device, dtype=param.dtype)
        for name, param in model.named_parameters()
        if param.requires_grad and name in init_state_dict
    }
    l2sp_numel = sum(param.numel() for param in source_params.values())

    best_metric = float("inf")
    best_epoch = 0
    best_state = mlp_base.clone_state_dict(model)
    history: List[Dict[str, float]] = []
    patience_left = int(early_stop_patience)
    val_losses: List[float] = []
    swa_states: List[Dict[str, torch.Tensor]] = []
    selection_mode = str(ft_selection_mode).strip().lower()
    if selection_mode not in {"raw_best", "smooth_best", "last_window_swa", "final"}:
        raise ValueError(f"Unsupported ft_selection_mode: {ft_selection_mode}")
    swa_start_epoch = max(1, int(epochs) - max(1, int(ft_swa_window)) + 1)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        running_target_loss = 0.0
        running_replay_loss = 0.0
        running_l2sp_loss = 0.0
        n_seen = 0
        replay_iter = iter(replay_loader) if use_replay and replay_loader is not None else None

        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            target_loss = loss_fn(pred, yb)
            total_loss = target_loss
            replay_loss_value = 0.0
            if use_replay and replay_iter is not None:
                try:
                    rb, ry = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    rb, ry = next(replay_iter)
                replay_loss = loss_fn(model(rb), ry)
                total_loss = total_loss + float(replay_loss_weight) * replay_loss
                replay_loss_value = float(replay_loss.item())
            l2sp_loss_value = 0.0
            if float(ft_l2sp_weight) > 0.0 and source_params and l2sp_numel > 0:
                l2sp_loss = sum(
                    torch.sum((param - source_params[name]) ** 2)
                    for name, param in model.named_parameters()
                    if param.requires_grad and name in source_params
                ) / float(l2sp_numel)
                total_loss = total_loss + float(ft_l2sp_weight) * l2sp_loss
                l2sp_loss_value = float(l2sp_loss.item())
            total_loss.backward()
            optimizer.step()

            running_loss += float(total_loss.item()) * len(xb)
            running_target_loss += float(target_loss.item()) * len(xb)
            running_replay_loss += replay_loss_value * len(xb)
            running_l2sp_loss += l2sp_loss_value * len(xb)
            n_seen += len(xb)

        train_loss = running_loss / n_seen if n_seen > 0 else float("nan")
        val_loss = mlp_base.evaluate_loss(model, val_part_df, x_cols, y_col, mu, sd, device, batch_size)
        val_losses.append(float(val_loss))
        selection_metric = rolling_median(val_losses, int(ft_smooth_window)) if selection_mode == "smooth_best" else float(val_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_target_loss": float(running_target_loss / n_seen if n_seen > 0 else float("nan")),
                "train_replay_loss": float(running_replay_loss / n_seen if n_seen > 0 else float("nan")),
                "train_l2sp_loss": float(running_l2sp_loss / n_seen if n_seen > 0 else float("nan")),
                "val_loss": float(val_loss),
                "selection_metric": float(selection_metric),
            }
        )
        if int(log_every) > 0 and (epoch == 1 or epoch % int(log_every) == 0):
            print(
                f"[LOSS] stage={stage_name} epoch={epoch} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} selection_metric={selection_metric:.6f}"
            )
        if selection_mode == "last_window_swa" and epoch >= swa_start_epoch:
            swa_states.append(mlp_base.clone_state_dict(model))
        if selection_mode == "final":
            best_metric = float(selection_metric)
            best_epoch = int(epoch)
            best_state = mlp_base.clone_state_dict(model)
        elif selection_mode != "last_window_swa" and selection_metric < best_metric:
            best_metric = float(selection_metric)
            best_epoch = int(epoch)
            best_state = mlp_base.clone_state_dict(model)
            patience_left = int(early_stop_patience)
        else:
            if epoch >= int(min_epochs_before_early_stop):
                patience_left -= 1
                if patience_left <= 0:
                    break

    if selection_mode == "last_window_swa":
        if not swa_states:
            swa_states.append(mlp_base.clone_state_dict(model))
        best_state = average_state_dicts(swa_states)
        best_epoch = int(history[-1]["epoch"]) if history else 0
        best_metric = rolling_median(val_losses, int(ft_smooth_window))

    model.load_state_dict(best_state)
    y_pred_train = mlp_base.predict_with_model(model, train_df, x_cols, mu, sd, device, batch_size)
    return {
        "model": model,
        "state_dict": best_state,
        "y_pred_train": y_pred_train,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "selection_mode": selection_mode,
        "ft_batch_mode": str(ft_batch_mode),
        "ft_smooth_window": int(ft_smooth_window),
        "ft_swa_window": int(ft_swa_window),
        "ft_l2sp_weight": float(ft_l2sp_weight),
        "used_validation": True,
        "freeze_hidden_layers": int(freeze_hidden_layers),
        "frozen_parameter_names": frozen_names,
        "train_cells": sorted(train_part_df["cell"].astype(str).unique().tolist(), key=week_runner.ridge_utils.cell_sort_key),
        "val_cells": sorted(val_part_df["cell"].astype(str).unique().tolist(), key=week_runner.ridge_utils.cell_sort_key),
        "replay_loss_weight": float(replay_loss_weight),
    }


def to_metrics(overall_row: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mae": float(overall_row.get("mae_mean", overall_row.get("mae", np.nan))),
        f"{prefix}_rmse": float(overall_row.get("rmse", np.nan)),
        f"{prefix}_mape_percent_mean": float(overall_row.get("mape_percent_mean", overall_row.get("mape_percent", np.nan))),
        f"{prefix}_smape_percent_mean": float(overall_row.get("smape_percent_mean", np.nan)),
        f"{prefix}_wmape_percent": float(overall_row.get("wmape_percent", np.nan)),
        f"{prefix}_r2": float(overall_row.get("r2", np.nan)),
    }


def save_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def build_model_from_state(
    *,
    state_dict: Dict[str, torch.Tensor],
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    activation: str,
    device: torch.device,
) -> torch.nn.Module:
    model = mlp_base.MLPRegressor(input_dim, hidden_dims, dropout=dropout, activation=activation).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Three-step runner for EOL70 / week5 source and transfer experiments.")
    ap.add_argument("--mode", type=str, choices=["source_only", "transfer"], default="source_only")
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--split_csv", type=str, required=True)
    ap.add_argument("--group_cond_csv", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--features", type=str, default="f1_w5,f3_w5,f5_w5,f8_w5,f2_w5")
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL70")
    ap.add_argument("--hidden_dims", type=str, default="64,16")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu"])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--ft_epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--ft_weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_cell_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=30)
    ap.add_argument("--min_epochs_before_early_stop", type=int, default=0)
    ap.add_argument("--ft_min_epochs_before_early_stop", type=int, default=0)
    ap.add_argument("--ft_freeze_hidden_layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--tail_q", type=float, default=0.95)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--source_checkpoint", type=str, default="")
    ap.add_argument("--target_support_ratio", type=float, default=1.0)
    ap.add_argument("--support_subset_mode", type=str, default="quantile", choices=["quantile", "random", "high_tail"])
    ap.add_argument("--support_subset_seed", type=int, default=17)
    ap.add_argument("--min_support_cells", type=int, default=6)
    ap.add_argument("--transfer_replay_weight", type=float, default=0.0)
    ap.add_argument("--min_target_val_cells", type=int, default=3)
    ap.add_argument("--ft_batch_mode", type=str, default="mini", choices=["mini", "full"])
    ap.add_argument("--ft_selection_mode", type=str, default="raw_best", choices=["raw_best", "smooth_best", "last_window_swa", "final"])
    ap.add_argument("--ft_smooth_window", type=int, default=25)
    ap.add_argument("--ft_swa_window", type=int, default=50)
    ap.add_argument("--ft_l2sp_weight", type=float, default=0.0)
    ap.add_argument("--evaluate_target_test", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    feature_aliases, x_cols, feature_week = week_runner.parse_feature_aliases(args.features)
    hidden_dims = mlp_base.parse_hidden_dims(args.hidden_dims)
    device = mlp_base.resolve_device(args.device)

    df = week_runner.load_lifetime_frame(Path(args.data_csv), x_cols, args.y_col, week_num=int(feature_week))
    cond_df = week_runner.ridge_utils.load_group_conditions(Path(args.group_cond_csv))
    df = week_runner.ridge_utils.add_condition_columns(df, cond_df)

    split_cells = week_runner.load_split_cells(Path(args.split_csv))
    train_cells = split_cells.get("train", [])
    ft_cells = split_cells.get("fine_tune", [])
    test_cells = split_cells.get("test", [])
    if not train_cells or not test_cells:
        raise ValueError("Split CSV must include non-empty train and test splits.")

    source_train_df = week_runner.mark_active_split(week_runner.select_cells(df, train_cells, "train"), "train", "Train")
    full_target_ft_df = (
        week_runner.mark_active_split(week_runner.select_cells(df, ft_cells, "fine_tune"), "fine_tune", "Fine-tune")
        if ft_cells
        else df.iloc[0:0].copy()
    )
    target_test_df = week_runner.mark_active_split(week_runner.select_cells(df, test_cells, "test"), "test", "Test")

    support_cells = []
    target_ft_df = full_target_ft_df
    if len(full_target_ft_df) > 0:
        support_cells = select_support_cells(
            full_target_ft_df,
            y_col=args.y_col,
            support_ratio=float(args.target_support_ratio),
            min_support_cells=int(args.min_support_cells),
            mode=str(args.support_subset_mode),
            seed=int(args.support_subset_seed),
        )
        target_ft_df = week_runner.mark_active_split(
            week_runner.select_cells(df, support_cells, "fine_tune_support"),
            "fine_tune",
            "Fine-tune",
        )

    source_checkpoint_path = Path(str(args.source_checkpoint)) if str(args.source_checkpoint).strip() else None
    if source_checkpoint_path is not None and source_checkpoint_path.exists():
        source_ckpt = torch.load(source_checkpoint_path, map_location="cpu")
        if source_ckpt["meta"]["features"] != list(feature_aliases):
            raise ValueError("Provided source checkpoint features do not match --features.")
        if list(source_ckpt["meta"]["hidden_dims"]) != list(hidden_dims):
            raise ValueError("Provided source checkpoint hidden_dims do not match --hidden_dims.")
        mu = np.asarray(source_ckpt["meta"]["mu"], dtype=float)
        sd = np.asarray(source_ckpt["meta"]["sd"], dtype=float)
        source_state_dict = source_ckpt["state_dict"]
        source_stage = {
            "state_dict": source_state_dict,
            "model": None,
            "train_cells": [],
            "val_cells": [],
            "y_pred_train": np.empty((0,), dtype=float),
        }
        source_metrics = source_ckpt["source_metrics"]
        source_checkpoint_out = source_checkpoint_path
    else:
        mu, sd = week_runner.ridge_utils.standardize_fit(source_train_df[list(x_cols)].to_numpy(dtype=float))
        source_stage = mlp_base.train_stage(
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
            require_validation=True,
        )
        source_train_pred_df = week_runner.ridge_utils.build_prediction_df(
            source_train_df,
            args.y_col,
            np.asarray(source_stage["y_pred_train"], dtype=float),
            "source_train",
        )
        source_inner_train_pred_df = week_runner.subset_prediction_df_by_cells(source_train_pred_df, source_stage["train_cells"], "source_inner_train")
        source_val_pred_df = week_runner.subset_prediction_df_by_cells(source_train_pred_df, source_stage["val_cells"], "source_val")
        source_inner_overall, _, _ = week_runner.summarize_prediction_split(source_inner_train_pred_df, "source_inner_train", float(args.tail_q))
        source_val_overall, _, _ = week_runner.summarize_prediction_split(source_val_pred_df, "source_val", float(args.tail_q))
        source_metrics = {
            **to_metrics(source_inner_overall.iloc[0].to_dict(), "source_inner_train"),
            **to_metrics(source_val_overall.iloc[0].to_dict(), "source_val"),
            "train_val_mae_gap": abs(float(source_inner_overall.iloc[0]["mae_mean"]) - float(source_val_overall.iloc[0]["mae_mean"])),
            "train_val_rmse_gap": abs(float(source_inner_overall.iloc[0]["rmse"]) - float(source_val_overall.iloc[0]["rmse"])),
            "train_val_mape_gap": abs(float(source_inner_overall.iloc[0]["mape_percent_mean"]) - float(source_val_overall.iloc[0]["mape_percent_mean"])),
        }
        source_checkpoint_out = out_root / "source_checkpoint.pt"
        save_checkpoint(
            source_checkpoint_out,
            {
                "meta": {
                    "features": list(feature_aliases),
                    "x_cols": list(x_cols),
                    "y_col": args.y_col,
                    "hidden_dims": list(hidden_dims),
                    "dropout": float(args.dropout),
                    "activation": args.activation,
                    "mu": np.asarray(mu, dtype=float).tolist(),
                    "sd": np.asarray(sd, dtype=float).tolist(),
                },
                "source_metrics": source_metrics,
                "state_dict": source_stage["state_dict"],
            },
        )

    trial_summary: Dict[str, object] = {
        "mode": args.mode,
        "features": ",".join(feature_aliases),
        "x_cols": list(x_cols),
        "feature_week": int(feature_week),
        "hidden_dims": ",".join(str(v) for v in hidden_dims),
        "dropout": float(args.dropout),
        "activation": args.activation,
        "epochs": int(args.epochs),
        "ft_epochs": int(args.ft_epochs),
        "lr": float(args.lr),
        "ft_lr": float(args.ft_lr),
        "weight_decay": float(args.weight_decay),
        "ft_weight_decay": float(args.ft_weight_decay),
        "ft_freeze_hidden_layers": int(args.ft_freeze_hidden_layers),
        "target_support_ratio": float(args.target_support_ratio),
        "transfer_replay_weight": float(args.transfer_replay_weight),
        "ft_batch_mode": str(args.ft_batch_mode),
        "ft_selection_mode": str(args.ft_selection_mode),
        "ft_smooth_window": int(args.ft_smooth_window),
        "ft_swa_window": int(args.ft_swa_window),
        "ft_l2sp_weight": float(args.ft_l2sp_weight),
        "support_cells": list(support_cells),
        "support_cell_count": int(len(support_cells)),
        "source_checkpoint": str(source_checkpoint_out),
        **source_metrics,
    }

    if args.mode == "transfer":
        if len(target_ft_df) == 0:
            raise ValueError("transfer mode requires a non-empty fine-tune subset.")
        try:
            active_stage = train_stage_with_replay(
                train_df=target_ft_df,
                replay_df=source_train_df,
                replay_loss_weight=float(args.transfer_replay_weight),
                x_cols=x_cols,
                y_col=args.y_col,
                mu=np.asarray(mu, dtype=float),
                sd=np.asarray(sd, dtype=float),
                hidden_dims=hidden_dims,
                dropout=float(args.dropout),
                activation=args.activation,
                epochs=int(args.ft_epochs),
                lr=float(args.ft_lr),
                weight_decay=float(args.ft_weight_decay),
                batch_size=int(args.batch_size),
                val_cell_frac=float(args.val_cell_frac),
                early_stop_patience=int(args.early_stop_patience),
                min_epochs_before_early_stop=int(args.ft_min_epochs_before_early_stop),
                seed=int(args.seed) + 1,
                device=device,
                init_state_dict=source_stage["state_dict"],
                freeze_hidden_layers=int(args.ft_freeze_hidden_layers),
                min_val_cells=int(args.min_target_val_cells),
                log_every=int(args.log_every),
                stage_name="target_finetune",
                ft_batch_mode=str(args.ft_batch_mode),
                ft_selection_mode=str(args.ft_selection_mode),
                ft_smooth_window=int(args.ft_smooth_window),
                ft_swa_window=int(args.ft_swa_window),
                ft_l2sp_weight=float(args.ft_l2sp_weight),
            )
        except ValueError as exc:
            invalid = {
                "mode": args.mode,
                "reason": str(exc),
                "target_support_ratio": float(args.target_support_ratio),
                "support_cell_count": int(len(support_cells)),
                "min_target_val_cells": int(args.min_target_val_cells),
            }
            save_json(out_root / "trial_invalid.json", invalid)
            raise
        target_ft_pred_df = week_runner.ridge_utils.build_prediction_df(
            target_ft_df,
            args.y_col,
            np.asarray(active_stage["y_pred_train"], dtype=float),
            "target_finetune",
        )
        target_ft_inner_pred_df = week_runner.subset_prediction_df_by_cells(
            target_ft_pred_df,
            active_stage["train_cells"],
            "target_finetune_inner_train",
        )
        target_ft_val_pred_df = week_runner.subset_prediction_df_by_cells(
            target_ft_pred_df,
            active_stage["val_cells"],
            "target_finetune_val",
        )
        target_ft_val_source_model = build_model_from_state(
            state_dict=source_stage["state_dict"],
            input_dim=len(x_cols),
            hidden_dims=hidden_dims,
            dropout=float(args.dropout),
            activation=args.activation,
            device=device,
        )
        target_ft_val_source_pred_df = week_runner.ridge_utils.build_prediction_df(
            target_ft_df[target_ft_df["cell"].astype(str).isin(active_stage["val_cells"])].copy(),
            args.y_col,
            mlp_base.predict_with_model(
                target_ft_val_source_model,
                target_ft_df[target_ft_df["cell"].astype(str).isin(active_stage["val_cells"])].copy(),
                x_cols,
                np.asarray(mu, dtype=float),
                np.asarray(sd, dtype=float),
                device,
                int(args.batch_size),
            ),
            "target_finetune_val_source_only",
        )
        target_ft_inner_overall, _, _ = week_runner.summarize_prediction_split(target_ft_inner_pred_df, "target_finetune_inner_train", float(args.tail_q))
        target_ft_val_overall, _, _ = week_runner.summarize_prediction_split(target_ft_val_pred_df, "target_finetune_val", float(args.tail_q))
        target_ft_val_source_overall, _, _ = week_runner.summarize_prediction_split(
            target_ft_val_source_pred_df,
            "target_finetune_val_source_only",
            float(args.tail_q),
        )
        ft_val_mae = float(target_ft_val_overall.iloc[0]["mae_mean"])
        source_val_mae = float(target_ft_val_source_overall.iloc[0]["mae_mean"])
        ft_val_mape = float(target_ft_val_overall.iloc[0]["mape_percent_mean"])
        source_val_mape = float(target_ft_val_source_overall.iloc[0]["mape_percent_mean"])
        trial_summary.update(
            {
                **to_metrics(target_ft_inner_overall.iloc[0].to_dict(), "target_ft_inner_train"),
                **to_metrics(target_ft_val_overall.iloc[0].to_dict(), "target_ft_val"),
                **to_metrics(target_ft_val_source_overall.iloc[0].to_dict(), "target_ft_val_source_only"),
                "target_ft_inner_train_cell_count": int(len(active_stage["train_cells"])),
                "target_ft_val_cell_count": int(len(active_stage["val_cells"])),
                "target_ft_train_val_mae_gap": abs(float(target_ft_inner_overall.iloc[0]["mae_mean"]) - float(target_ft_val_overall.iloc[0]["mae_mean"])),
                "target_ft_val_vs_source_only_mae_improve_percent": (source_val_mae - ft_val_mae) / source_val_mae * 100.0 if source_val_mae > 0 else float("nan"),
                "target_ft_val_vs_source_only_mape_improve_percent": (source_val_mape - ft_val_mape) / source_val_mape * 100.0 if source_val_mape > 0 else float("nan"),
            }
        )
        save_checkpoint(
            out_root / "finetuned_checkpoint.pt",
            {
                "meta": {
                    "source_checkpoint": str(source_checkpoint_out),
                    "features": list(feature_aliases),
                    "x_cols": list(x_cols),
                    "y_col": args.y_col,
                    "hidden_dims": list(hidden_dims),
                    "mu": np.asarray(mu, dtype=float).tolist(),
                    "sd": np.asarray(sd, dtype=float).tolist(),
                },
                "trial_summary": trial_summary,
                "state_dict": active_stage["state_dict"],
            },
        )
        target_ft_pred_df.to_csv(out_root / "predictions_target_finetune.csv", index=False)
        target_ft_val_source_pred_df.to_csv(out_root / "predictions_target_finetune_val_source_only.csv", index=False)
        target_ft_inner_overall.to_csv(out_root / "target_finetune_inner_train_overall_metrics.csv", index=False)
        target_ft_val_overall.to_csv(out_root / "target_finetune_val_overall_metrics.csv", index=False)
        target_ft_val_source_overall.to_csv(out_root / "target_finetune_val_source_only_overall_metrics.csv", index=False)

        if args.evaluate_target_test:
            target_test_pred_df = week_runner.ridge_utils.build_prediction_df(
                target_test_df,
                args.y_col,
                mlp_base.predict_with_model(active_stage["model"], target_test_df, x_cols, np.asarray(mu, dtype=float), np.asarray(sd, dtype=float), device, int(args.batch_size)),
                "target_test_finetuned",
            )
            target_test_overall = week_runner.ridge_utils.summarize_overall(target_test_pred_df, "target_test_finetuned", float(args.tail_q))
            trial_summary.update(to_metrics(target_test_overall.iloc[0].to_dict(), "target_test"))
            target_test_pred_df.to_csv(out_root / "predictions_target_test.csv", index=False)
            target_test_overall.to_csv(out_root / "target_test_overall_metrics.csv", index=False)

    save_json(out_root / "trial_summary.json", trial_summary)
    print(json.dumps(trial_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
