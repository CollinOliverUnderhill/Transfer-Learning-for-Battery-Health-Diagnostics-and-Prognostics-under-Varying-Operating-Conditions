# Week-Based External Code Dependencies

This folder collects the code dependencies that the week-based IVAS workflow
still uses from outside these three main folders:

- `E:\Datasets\IVAS\Data`
- `E:\Datasets\IVAS\Processing_Data_dd_exclude`
- `E:\Datasets\IVAS\Lifetime_RUL_prediction`

Copied code:

- `SOHest/MLP_codes/domain_train_test_by_group_mlp.py`
- `SOHest/Ridge_codes/domain_train_test_by_group.py`
- `Feature_extraction/extract_ivas_lifetime_10features.py`
- `Feature_engineering/analyze_feature_lifetime_correlations.py`

Why these were copied:

- `Lifetime_RUL_prediction/EOLxx/codes/run_lifetime_transfer_mlp.py`
  dynamically loads the shared MLP base script, which in turn depends on the
  shared ridge utility script.
- `Lifetime_RUL_prediction/EOLxx/codes/extract_ivas_lifetime_multiweek_and_augment_eol.py`
  imports `extract_ivas_lifetime_10features.py` as its base feature builder.
- `Lifetime_RUL_prediction/EOLxx/codes/analyze_feature_lifetime_correlations_multiweek.py`
  imports `analyze_feature_lifetime_correlations.py` as its base analysis
  module.

Still not moved here:

- Data files such as `Groupcondi.csv`, `Valid_cells.csv`
- Raw upstream data folders such as `capacity_fade`, `Q_interpolated`,
  `RPT_json`, `Cycling_json`
- Author reference CSVs under
  `Codes/tingkai-li-early-prediction-varying-usage-data-d1f5535/feature_extraction`

Notes:

- The copied scripts keep their original hard-coded data paths.
- The `SOHest` subfolder structure is preserved so the copied MLP script can
  still find its ridge dependency through its relative path logic.
