param(
  [int[]]$Weeks = @(6, 7, 8, 9, 10),
  [switch]$Cpu,
  [string]$PythonExe = '',
  [string]$OutRoot = 'E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\outputs\fixed_w6_10_from_stage3_final_rerun_400_legacy400'
)

$ErrorActionPreference = 'Stop'

if ($PythonExe.Trim().Length -gt 0) {
  $py = $PythonExe
} elseif ($env:CONDA_PREFIX -and (Test-Path -LiteralPath (Join-Path $env:CONDA_PREFIX 'python.exe'))) {
  $py = Join-Path $env:CONDA_PREFIX 'python.exe'
} else {
  $py = 'D:\Anaconda\envs\torchenv\python.exe'
}

$root3 = 'E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step'
$data = 'E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\features\feature_table_all_cells_multiweek_EOL70.csv'
$group = 'E:\Datasets\IVAS\Groupcondi.csv'
$split = 'E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\domain_split\cell_split_targetspread_w5_EOL70.csv'
$week5Reference = 'E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\outputs\stage3_final_rerun_400'

if ($split -match 'High_tail|random') {
  throw "Refusing to run fixed rerun comparison with non-fixed/high-tail split: $split"
}

foreach ($path in @($py, $data, $group, $split, $week5Reference)) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Required path does not exist: $path"
  }
}

$env:PYTHONWARNINGS = 'ignore'
if ($Cpu) {
  $env:CUDA_VISIBLE_DEVICES = ''
} else {
  $env:CUDA_VISIBLE_DEVICES = '0'
}
$env:CUDA_MODULE_LOADING = 'LAZY'

function Invoke-PythonChecked {
  param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
  )

  $oldErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    & $py @Arguments
    $exitCode = $LASTEXITCODE
  } catch {
    throw "Failed to start Python command: $py $($Arguments -join ' ')`n$($_.Exception.Message)"
  } finally {
    $ErrorActionPreference = $oldErrorActionPreference
  }

  if ($null -eq $exitCode) {
    Write-Warning "Python command returned no LASTEXITCODE in PowerShell; assuming success and continuing: $py $($Arguments -join ' ')"
    return
  }

  if ($exitCode -ne 0) {
    throw "Python command failed with exit code $exitCode`: $py $($Arguments -join ' ')"
  }
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$notePath = Join-Path $OutRoot 'protocol_note.txt'
@(
  'Fixed week6-10 comparison from stage3_final_rerun_400 legacy400 protocol.',
  "Week5 reference: $week5Reference",
  "Data CSV: $data",
  "Split CSV: $split",
  'Design: same ordinary week5 fixed train/fine_tune/test cell split; week-specific features/candidates per week.',
  'No target test metrics are used in Stage1 or Stage2 optimization; test is only evaluated in Stage3 final evaluation.',
  "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
) | Set-Content -LiteralPath $notePath -Encoding UTF8

$weekListText = (($Weeks | ForEach-Object { "week$_" }) -join ', ')
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Requested weeks: $weekListText"
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Python executable: $py"
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] CUDA_VISIBLE_DEVICES: '$env:CUDA_VISIBLE_DEVICES'"
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Output root: $OutRoot"
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Fixed split: $split"

foreach ($week in $Weeks) {
  if ($week -lt 6 -or $week -gt 10) {
    throw "This fixed comparison script is intended for weeks 6-10. Got: $week"
  }

  $candidate = "E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\features\informed_feature_candidates_w${week}_EOL70.csv"
  if (-not (Test-Path -LiteralPath $candidate)) {
    throw "Missing candidate CSV for week$week`: $candidate"
  }

  $outWeek = Join-Path $OutRoot ("week{0}" -f $week)
  New-Item -ItemType Directory -Force -Path $outWeek | Out-Null
  Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ===== week$week fixed legacy400 Stage1/2/3 ====="
  Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Candidate CSV: $candidate"

  if (-not (Test-Path -LiteralPath (Join-Path $outWeek 'stage1\stage1_top_source_checkpoints.csv'))) {
    Invoke-PythonChecked "$root3\stage1_source_search_optuna_legacy.py" `
      --python_exe $py `
      --data_csv $data `
      --split_csv $split `
      --group_cond_csv $group `
      --candidate_csv $candidate `
      --out_dir "$outWeek\stage1" `
      --study_name "stage1_w${week}_fixed_from_rerun400_legacy400" `
      --n_trials 80 `
      --top_k 5 `
      --max_feature_candidates 20 `
      --width_candidates "8,16,32,64" `
      --max_depth 4 `
      --y_col "lifetime_weeks_EOL70" `
      --epochs 800 `
      --batch_size 16 `
      --val_cell_frac 0.2 `
      --early_stop_patience 25 `
      --min_epochs_before_early_stop 400 `
      --lr_min 0.0001 `
      --lr_max 0.003 `
      --weight_decay_min 1e-7 `
      --weight_decay_max 0.001 `
      --seed 2
  } else {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] week$week Stage1 exists, skip."
  }

  if (-not (Test-Path -LiteralPath (Join-Path $outWeek 'stage2\stage2_best_configs.csv'))) {
    Invoke-PythonChecked "$root3\stage2_finetune_search_optuna_legacy.py" `
      --python_exe $py `
      --data_csv $data `
      --split_csv $split `
      --group_cond_csv $group `
      --stage1_top_csv "$outWeek\stage1\stage1_top_source_checkpoints.csv" `
      --out_dir "$outWeek\stage2" `
      --study_name "stage2_w${week}_fixed_from_rerun400_legacy400" `
      --n_trials 80 `
      --top_k 5 `
      --y_col "lifetime_weeks_EOL70" `
      --dropout 0.0 `
      --activation "relu" `
      --batch_size 16 `
      --val_cell_frac 0.2 `
      --early_stop_patience 25 `
      --ft_min_epochs_before_early_stop 400 `
      --min_target_val_cells 3 `
      --seed 19 `
      --support_ratios "0.67,1.0" `
      --ft_epoch_choices "400,800" `
      --replay_weight_choices "0.0,0.1,0.3,1.0" `
      --ft_freeze_hidden_layers_min 2 `
      --ft_lr_min 1e-5 `
      --ft_lr_max 0.01 `
      --ft_weight_decay_min 1e-7 `
      --ft_weight_decay_max 0.001
  } else {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] week$week Stage2 exists, skip."
  }

  if (-not (Test-Path -LiteralPath (Join-Path $outWeek 'stage3_final\stage3_final_report.json'))) {
    Invoke-PythonChecked "$root3\stage3_final_evaluate_legacy.py" `
      --python_exe $py `
      --data_csv $data `
      --split_csv $split `
      --group_cond_csv $group `
      --stage2_best_csv "$outWeek\stage2\stage2_best_configs.csv" `
      --out_dir "$outWeek\stage3_final" `
      --y_col "lifetime_weeks_EOL70" `
      --dropout 0.0 `
      --activation "relu" `
      --batch_size 16 `
      --val_cell_frac 0.2 `
      --early_stop_patience 25 `
      --min_source_epochs 400 `
      --min_ft_epochs 400 `
      --min_epochs_before_early_stop 400 `
      --ft_min_epochs_before_early_stop 400 `
      --support_subset_mode "quantile" `
      --support_subset_seed 17 `
      --min_support_cells 6 `
      --min_target_val_cells 3 `
      --seed 101
  } else {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] week$week Stage3 exists, skip."
  }

  Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ===== week$week done ====="
}

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] All requested fixed weeks finished."
