# EOL70 RUL Workspace

This folder is a self-contained workspace for the revised EOL70 setup.

- `codes/`
  - split generation with target-domain fine-tune spread
  - fixed transfer-model config copied from the EOL60 baseline
  - training runner used for the final experiments
- `domain_split/`
  - week-specific split files
  - group-level split summaries
  - manifest for all generated split files
- `features/`
  - multi-week feature table for EOL70
  - original exclude-based augmented split table
  - week availability summary
- `feature_engineering/`
  - week-specific correlation csv / md / svg / html
  - scatter plot folders
- `results/`
  - actual benchmark / transfer-model outputs for `w3 / w5 / w10 / w15`
  - comparison csv for the repartitioned experiments

Current split logic:
- source/train groups are taken from the lower-lifetime side after sorting by group lifetime median
- target groups are the remaining groups
- `fine_tune` cells are sampled inside the target groups so they are spread across the target domain
- `test` cells are the remaining target-domain cells

Current model config:
- temporarily fixed to the validated EOL60 baseline (`8x8x8 / relu / lr=7e-4 / wd=1e-4 / bs=16 / ft_epochs=800 / freeze_hidden_layers=2`)
- stored explicitly in `codes/fixed_transfer_config.json` so EOL-specific tuning can replace it later without changing other folders
