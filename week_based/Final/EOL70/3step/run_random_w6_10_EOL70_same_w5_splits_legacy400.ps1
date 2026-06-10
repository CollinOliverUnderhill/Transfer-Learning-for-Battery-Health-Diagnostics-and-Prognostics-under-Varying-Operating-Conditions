param(
  [object[]]$Seeds = (0..19),
  [object[]]$Weeks = (6..10),
  [switch]$Cpu,
  [string]$PythonExe = '',
  [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'

function Convert-IntegerList {
  param(
    [object[]]$RawValues,
    [string]$Name
  )

  $parsed = New-Object System.Collections.Generic.List[int]
  foreach ($item in $RawValues) {
    if ($null -eq $item) {
      continue
    }

    foreach ($part in ([string]$item -split ',')) {
      $trimmed = $part.Trim()
      if ($trimmed.Length -eq 0) {
        continue
      }
      $parsed.Add([int]$trimmed)
    }
  }

  if ($parsed.Count -eq 0) {
    throw "No $Name specified."
  }

  return [int[]]$parsed.ToArray()
}

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

$Seeds = Convert-IntegerList $Seeds 'seeds'
$Weeks = Convert-IntegerList $Weeks 'weeks'

if ($PythonExe.Trim().Length -gt 0) {
  $py = $PythonExe
} else {
  $py = 'D:\Anaconda\envs\torchenv\python.exe'
}

$root3 = 'E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step'
$data = 'E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\features\feature_table_all_cells_multiweek_EOL70.csv'
$group = 'E:\Datasets\IVAS\Groupcondi.csv'
$splitRoot = 'E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\domain_split'
if ($OutputRoot.Trim().Length -gt 0) {
  $outRoot = $OutputRoot
} else {
  $outRoot = 'E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\3step\outputs\random_w6_10_EOL70_same_w5_splits_legacy400'
}

$env:PYTHONWARNINGS = 'ignore'
if ($Cpu) {
  $env:CUDA_VISIBLE_DEVICES = ''
} else {
  $env:CUDA_VISIBLE_DEVICES = '0'
}
$env:CUDA_MODULE_LOADING = 'LAZY'

New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$seedListText = (($Seeds | ForEach-Object { 'seed{0:D3}' -f $_ }) -join ', ')
$weekListText = (($Weeks | ForEach-Object { 'w{0}' -f $_ }) -join ', ')
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Requested seeds: $seedListText"
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Requested weeks: $weekListText"
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Python executable: $py"
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Output root: $outRoot"
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] CUDA_VISIBLE_DEVICES: '$env:CUDA_VISIBLE_DEVICES'"

foreach ($seed in $Seeds) {
  $seedName = ('seed{0:D3}' -f $seed)
  $split = Join-Path $splitRoot "w5_EOL70_random_$seedName\cell_split_targetrandom_w5_EOL70_$seedName.csv"

  if (-not (Test-Path -LiteralPath $split)) {
    throw "Missing ordinary week5 random split csv for $seedName`: $split"
  }
  if ($split -like '*High_tail*') {
    throw "Refusing to use a High_tail split: $split"
  }

  foreach ($week in $Weeks) {
    if ($week -lt 6 -or $week -gt 10) {
      throw "This companion run is intended for weeks 6..10, got week $week."
    }

    $candidate = "E:\Datasets\IVAS\Codes\chunqiu_codes\week_based\Final\EOL70\features\informed_feature_candidates_w${week}_EOL70.csv"
    if (-not (Test-Path -LiteralPath $candidate)) {
      throw "Missing feature candidate csv for week $week`: $candidate"
    }

    $outWeek = Join-Path (Join-Path $outRoot $seedName) "week$week"
    New-Item -ItemType Directory -Force -Path $outWeek | Out-Null
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ===== $seedName week$week ordinary-split legacy_400 Stage1/2/3 ====="

    if (-not (Test-Path -LiteralPath (Join-Path $outWeek 'stage1\stage1_top_source_checkpoints.csv'))) {
      Invoke-PythonChecked "$root3\stage1_source_search_optuna_legacy.py" `
        --data_csv $data `
        --split_csv $split `
        --group_cond_csv $group `
        --candidate_csv $candidate `
        --out_dir "$outWeek\stage1" `
        --study_name "stage1_w${week}_random_${seedName}_legacy400" `
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
      Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $seedName week$week Stage1 exists, skip."
    }

    if (-not (Test-Path -LiteralPath (Join-Path $outWeek 'stage2\stage2_best_configs.csv'))) {
      Invoke-PythonChecked "$root3\stage2_finetune_search_optuna_legacy.py" `
        --data_csv $data `
        --split_csv $split `
        --group_cond_csv $group `
        --stage1_top_csv "$outWeek\stage1\stage1_top_source_checkpoints.csv" `
        --out_dir "$outWeek\stage2" `
        --study_name "stage2_w${week}_random_${seedName}_legacy400" `
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
      Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $seedName week$week Stage2 exists, skip."
    }

    if (-not (Test-Path -LiteralPath (Join-Path $outWeek 'stage3_final\stage3_final_report.json'))) {
      Invoke-PythonChecked "$root3\stage3_final_evaluate_legacy.py" `
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
      Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $seedName week$week Stage3 exists, skip."
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ===== $seedName week$week done ====="
  }
}

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] All requested week6-10 ordinary-split runs finished."
