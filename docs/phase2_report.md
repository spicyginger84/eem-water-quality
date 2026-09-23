# Phase 2 — Separate chemistry from drift

**Status:** rerun completed, 2026-09-23.

This report tests whether the EEM representation transfers to an unseen station,
an unseen month, or an unseen station-month combination. The rerun uses the
processed, calibrated EEM data and the `EEMpca` representation with 30 PCA
components. The zero-column mask, scaling and PCA are fitted inside each
training fold.

The diagnostics compare a global-mean baseline, a station-mean baseline,
linear regression (LR) and XGBoost. For an unseen station, the station baseline
falls back to the global mean. No nested 5-fold model selection is used here:
all models use the same folds and input representation, and are fitted directly
for this fast diagnostic. Adding model-selection CV to only some runs would
change the training protocol and make the experiment comparison less
symmetric; a later CV study should apply exactly the same selection procedure
to every model and protocol.

## Commands

```bash
PYTHONPATH=src python scripts/evaluate_holdouts.py \
  --data data/processed --output runs/phase2_station_loso_eempca_v3 \
  --protocol loso --group-col Point \
  --targets BOD COD TOC BOD_COD \
  --experiments EEMpca --models linear xgboost \
  --pca-components 30 --seed 42 --n-jobs 1

PYTHONPATH=src python scripts/evaluate_holdouts.py \
  --data data/processed --output runs/phase2_month_loso_eempca_v3 \
  --protocol loso --group-col Month \
  --targets BOD COD TOC BOD_COD \
  --experiments EEMpca --models linear xgboost \
  --pca-components 30 --seed 42 --n-jobs 1

PYTHONPATH=src python scripts/evaluate_holdouts.py \
  --data data/processed --output runs/phase2_two_way_eempca_v3 \
  --protocol two_way --group-col Point --secondary-group-col Month \
  --targets BOD COD TOC BOD_COD \
  --experiments EEMpca --models linear xgboost \
  --pca-components 30 --seed 42 --n-jobs 1
```

## How to read the R² values

Each LOSO/LOMO cell is `arithmetic mean ± SD / pooled R²`.

- **Arithmetic mean** gives every held-out group the same weight. It is the
  mean of the per-station or per-month R² values.
- **Pooled R²** concatenates all held-out predictions and observations before
  computing one R². It is therefore weighted by the number of observations in
  each group.

The station protocol has 43 valid station folds and the month protocol has 12
month folds. For the two-way protocol, seed 42 gives 385 training rows, 85
validation rows, 32 untouched test rows and 258 excluded rows; the table shows
the single test-set R².

## Results

### Leave-one-station-out (LOSO)

| Target | Global mean | Station mean | EEMpca + LR | EEMpca + XGB |
|---|---:|---:|---:|---:|
| BOD | -11.012 ± 18.607 / -0.030 | -11.012 ± 18.607 / -0.030 | -6.148 ± 21.615 / -0.439 | -7.250 ± 36.613 / -0.202 |
| COD | -18.119 ± 37.995 / -0.033 | -18.119 ± 37.995 / -0.033 | -2.962 ± 8.414 / 0.191 | -3.404 ± 10.979 / 0.200 |
| TOC | -13.401 ± 23.393 / -0.030 | -13.401 ± 23.393 / -0.030 | -3.269 ± 11.731 / 0.032 | -3.298 ± 12.113 / 0.218 |
| BOD_COD | -1.065 ± 1.660 / -0.024 | -1.065 ± 1.660 / -0.024 | -2.251 ± 7.072 / -0.434 | -1.356 ± 5.111 / -0.037 |

### Leave-one-month-out (LOMO)

| Target | Global mean | Station mean | EEMpca + LR | EEMpca + XGB |
|---|---:|---:|---:|---:|
| BOD | -0.083 ± 0.133 / -0.007 | 0.248 ± 0.376 / 0.422 | -0.043 ± 0.710 / 0.202 | 0.232 ± 0.291 / 0.370 |
| COD | -0.153 ± 0.199 / -0.014 | 0.328 ± 0.414 / 0.488 | 0.321 ± 0.509 / 0.501 | 0.447 ± 0.171 / 0.511 |
| TOC | -0.128 ± 0.137 / -0.014 | 0.389 ± 0.331 / 0.461 | 0.319 ± 0.443 / 0.437 | 0.410 ± 0.392 / 0.482 |
| BOD_COD | -0.061 ± 0.061 / -0.009 | 0.226 ± 0.283 / 0.308 | -0.151 ± 0.536 / -0.015 | -0.040 ± 0.418 / 0.092 |

### Two-way holdout: unseen station and unseen month

| Target | Global mean | Station mean | EEMpca + LR | EEMpca + XGB |
|---|---:|---:|---:|---:|
| BOD | -0.208 | -0.208 | -5.784 | -1.188 |
| COD | -0.221 | -0.221 | -4.287 | -2.319 |
| TOC | -0.282 | -0.282 | -9.881 | -4.992 |
| BOD_COD | -0.049 | -0.049 | -7.986 | -1.929 |

## Interpretation

1. **Station transfer is weak.** Every LOSO arithmetic mean is negative. The
   pooled R² is positive for COD and TOC (0.191/0.200 and 0.032/0.218 for
   LR/XGB), but remains negative for BOD and BOD_COD. The large difference
   between the arithmetic and pooled values shows that a few small or difficult
   stations strongly affect the unweighted mean.

2. **Month transfer is stronger.** LOMO pooled R² is positive for COD and TOC
   (0.501/0.511 and 0.437/0.482 for LR/XGB). XGB also gives positive pooled R²
   for BOD (0.370), while the station baseline is stronger for BOD and
   BOD_COD. The station baseline remains a useful reference because it captures
   station-specific target levels without EEM features.

3. **The strict two-way test fails.** All EEMpca R² values are negative when
   both station and month are unseen. The best result is XGB for BOD
   (−1.188), still below the global baseline. The current representation does
   not yet support reliable transfer to an unseen station-month combination.

4. **Station and month effects differ.** The stronger LOMO results than LOSO
   results indicate that station-specific chemistry or station fingerprints are
   a major source of error. Per-station errors and the month effect should be
   inspected to distinguish seasonal chemistry from instrument or session drift.

## Limitations and next steps

- These are fast diagnostic runs without nested 5-fold model selection. LR and
  XGB use identical folds and are evaluated directly; no model is selected from
  held-out rows.
- Both R² summaries should be retained: arithmetic mean treats groups equally,
  while pooled R² treats observations equally. They answer different questions
  and can diverge substantially in this dataset.
- The next diagnostic is to inspect per-station errors and determine whether the
  month effect reflects seasonal chemistry, instrument drift or session effects.

## Artifacts

- `runs/phase2_station_loso_eempca_v3/holdout_metrics.csv`
- `runs/phase2_month_loso_eempca_v3/holdout_metrics.csv`
- `runs/phase2_two_way_eempca_v3/holdout_metrics.csv`
- Each run also contains `holdout_predictions.csv`, `pca_mask_summary.csv` and
  `config.json`.
