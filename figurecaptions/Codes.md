# Key Code Snippets for the Three-Step RUL Pipeline

This note summarizes the key implementation logic for the three main stages of the RUL pipeline: source-domain training, target-domain fine-tuning, and target-domain test evaluation. The snippets are shortened for readability and keep only the core operations.

## 1. Source Training Stage

Purpose: train the source-domain neural network using the source training cells, fit feature standardization on the source domain, and save the trained source checkpoint for later fine-tuning.

```python
mu, sd = week_runner.ridge_utils.standardize_fit(
    source_train_df[list(x_cols)].to_numpy(dtype=float)
)

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
    seed=int(args.seed),
    device=device,
    freeze_hidden_layers=0,
    stage_name="source_train",
    require_validation=True,
)

save_checkpoint(
    out_root / "source_checkpoint.pt",
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
```

Brief explanation: the model first learns a source-domain mapping from early-life features to RUL. The fitted normalization parameters and trained network weights are saved as `source_checkpoint.pt`.

## 2. Target Fine-Tuning Stage

Purpose: initialize the model from the source checkpoint and update it using the selected target fine-tuning cells. Optional source replay and layer freezing are used to control overfitting and preserve useful source-domain representations.

```python
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
    seed=int(args.seed) + 1,
    device=device,
    init_state_dict=source_stage["state_dict"],
    freeze_hidden_layers=int(args.ft_freeze_hidden_layers),
    min_val_cells=int(args.min_target_val_cells),
    stage_name="target_finetune",
    ft_batch_mode=str(args.ft_batch_mode),
    ft_selection_mode=str(args.ft_selection_mode),
    ft_l2sp_weight=float(args.ft_l2sp_weight),
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
```

Brief explanation: fine-tuning starts from the source-trained weights, updates the trainable layers on target support cells, and saves the adapted model as `finetuned_checkpoint.pt`.

The core optimization loop inside `train_stage_with_replay` is:

```python
for xb, yb in train_loader:
    optimizer.zero_grad(set_to_none=True)
    pred = model(xb)
    target_loss = loss_fn(pred, yb)
    total_loss = target_loss

    if use_replay and replay_iter is not None:
        rb, ry = next(replay_iter)
        replay_loss = loss_fn(model(rb), ry)
        total_loss = total_loss + float(replay_loss_weight) * replay_loss

    if float(ft_l2sp_weight) > 0.0:
        l2sp_loss = sum(
            torch.sum((param - source_params[name]) ** 2)
            for name, param in model.named_parameters()
            if param.requires_grad and name in source_params
        ) / float(l2sp_numel)
        total_loss = total_loss + float(ft_l2sp_weight) * l2sp_loss

    total_loss.backward()
    optimizer.step()
```

Brief explanation: the fine-tuning loss is mainly the target-cell regression loss, optionally combined with source replay loss and L2-SP regularization.

## 3. Target Test Stage

Purpose: evaluate the fine-tuned model on held-out target test cells and export prediction files and overall metrics.

```python
if args.evaluate_target_test:
    target_test_pred_df = week_runner.ridge_utils.build_prediction_df(
        target_test_df,
        args.y_col,
        mlp_base.predict_with_model(
            active_stage["model"],
            target_test_df,
            x_cols,
            np.asarray(mu, dtype=float),
            np.asarray(sd, dtype=float),
            device,
            int(args.batch_size),
        ),
        "target_test_finetuned",
    )

    target_test_overall = week_runner.ridge_utils.summarize_overall(
        target_test_pred_df,
        "target_test_finetuned",
        float(args.tail_q),
    )

    trial_summary.update(
        to_metrics(target_test_overall.iloc[0].to_dict(), "target_test")
    )
    target_test_pred_df.to_csv(out_root / "predictions_target_test.csv", index=False)
    target_test_overall.to_csv(out_root / "target_test_overall_metrics.csv", index=False)
```

Brief explanation: the adapted model predicts RUL for target test cells that were not used for fine-tuning. The code then summarizes test MAE, RMSE, MAPE, and related metrics.

The final evaluation script reads the exported test metrics and stores the final report:

```python
transfer_test = read_single_row_csv(
    out_dir / "transfer_model" / "test_overall_metrics.csv"
)
benchmark_test = read_single_row_csv(
    out_dir / "benchmark" / "test_overall_metrics.csv"
)

summary_report = {
    "benchmark_test_overall": benchmark_test,
    "transfer_test_overall": transfer_test,
    "selected_stage2_row": best_row,
    "selected_stage1_summary": source_summary,
    "final_package_root": str(out_dir),
}

save_json(out_dir / "stage3_final_report.json", summary_report)
```

Brief explanation: this creates the final experiment summary by collecting benchmark and transfer-learning test results into `stage3_final_report.json`.
