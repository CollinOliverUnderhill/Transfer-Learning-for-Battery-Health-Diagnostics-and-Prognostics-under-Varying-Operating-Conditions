# EOL60 RUL Workspace

This folder is a self-contained workspace for the revised EOL60 setup.

- `codes/`
  - split generation with target-domain fine-tune spread
  - feature extraction / feature engineering scripts used by this EOL
  - training runner used for the final experiments
- `domain_split/`
  - week-specific split files
  - group-level split summaries
  - manifest for all generated split files
- `features/`
  - multi-week feature table for EOL60
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
