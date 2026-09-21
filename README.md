# EEM Water-Quality Prediction

This project predicts laboratory water-quality parameters from excitation–emission matrix (EEM) fluorescence measurements. The main targets are BOD, COD, TOC and the derived BOD/COD ratio. Models can use EEM features alone, EEM features combined with laboratory covariates, or tabular covariates alone.

## Data

The raw data is under `data/2026ER_data_for Viet`:

- Monthly folders contain one EEM workbook per sample.
- `20260120_ER_data_HS.xlsx`, sheet `Data`, contains laboratory parameters with `Label`, `Month` and `Round`.
- The `Location` sheet contains station metadata.
- The workbook records sampling year 2023 even though the directory name contains `2026ER`.

The processed dataset currently contains 760 aligned samples from 45 stations. Each EEM has 271 excitation values from 280–550 nm and 57 emission values from 220–500 nm, so a flattened EEM contains 15,447 features.

The processed files are:

```text
data/processed/
├── eem.npy                  EEM array with shape (760, 271, 57)
├── samples.parquet          aligned metadata and laboratory parameters
├── samples.csv              CSV copy of samples.parquet
├── wavelengths.npz          excitation and emission wavelengths
├── eem_mapping.csv          mapping decision for every raw EEM file
├── locations.parquet        station metadata
├── locations.csv            CSV copy of locations.parquet
└── processing_summary.json  processing configuration and counts
```

## Processing and mapping pipeline

Rebuild the processed data from the raw files with:

```bash
PYTHONPATH=src .venv/bin/python scripts/process_raw_data.py \
  --raw 'data/2026ER_data_for Viet' \
  --output data/processed
```

The mapper uses `Label + Month + Round` as the sample key. A filename with a round, such as `ha1-1_3.xlsx`, is interpreted as station `Ha1`, round `1`, in month `3`. A filename containing only a station is accepted only when exactly one laboratory row exists for that station and month. Ambiguous files such as `ha20_3.xlsx` are recorded as `unmatched` instead of being assigned to an arbitrary round.

Each EEM workbook is checked for a numeric wavelength header, a consistent matrix shape, finite values and the common wavelength grid. The default completeness filter requires numeric BOD, COD, TOC, SS and EC, treats `ND` and blanks as missing, and excludes samples with `Depth > 2 m`. Missing depth is retained.

The current mapping contains 760 accepted samples, 239 files without a unique laboratory match and 2 files excluded by the depth limit. `eem_mapping.csv` records the source filename, laboratory row, matching method and exclusion reason.

Inspect the processed data and list the available feature experiments:

```bash
PYTHONPATH=src .venv/bin/python -m eem_water_quality inspect --data data/processed
PYTHONPATH=src .venv/bin/python -m eem_water_quality list
```

## Feature representations

The classical pipeline builds features using the training partition only:

- `TOC`, `SS` and `EC`: numeric tabular covariates with median imputation and standard scaling.
- `EEM`: flattened original EEM values (15,447 features).
- `EEMpca`: standardized EEM followed by PCA; the default is 30 components.
- `PARAFAC`: rank-4 PARAFAC scores by default. The standalone `parafac` command uses a nonnegative PARAFAC model.
- Combined experiments concatenate EEM/PCA/PARAFAC features with selected tabular covariates.

The target column is never used as a predictor. In particular, every
experiment that declares `TOC` as a covariate is skipped when the target is
TOC, including `TOC_SS_EC`, `EEMpca_TOC` and `EEMpca_TOC_SS_EC`. The skipped
experiment is recorded in that target's `validation_metrics.csv`.

The experiment catalog includes `TOC`, `TOC_SS_EC`, all `EEM`, `EEMpca` and `PARAFAC` variants with optional `TOC` or `SS_EC` covariates, and their combined forms.

## Classical models

Available models are:

- `linear`: Linear Regression
- `tree`: Decision Tree Regression
- `svr`: standardized RBF Support Vector Regression
- `xgboost`: XGBoost Regression

Example using PCA EEM and tabular baselines:

```bash
PYTHONPATH=src .venv/bin/python -m eem_water_quality ml \
  --data data/processed \
  --output runs/ml_bod_cod_toc \
  --targets BOD COD TOC \
  --split random --seed 42 \
  --models linear xgboost \
  --experiments TOC TOC_SS_EC EEMpca EEMpca_TOC EEMpca_SS_EC \
  --pca-components 30 --n-jobs 1
```

The selected four-target benchmark was run with:

```bash
PYTHONPATH=src .venv/bin/python -m eem_water_quality ml \
  --data data/processed \
  --output runs/selected_4targets_lr_xgb_no_leakage \
  --targets BOD COD TOC BOD_COD \
  --split random --seed 42 \
  --models linear xgboost \
  --experiments TOC TOC_SS_EC EEMpca EEMpca_TOC EEMpca_SS_EC EEMpca_TOC_SS_EC \
  --pca-components 30 --n-jobs 1
```

The standalone nonnegative PARAFAC + SVR command is:

```bash
PYTHONPATH=src .venv/bin/python -m eem_water_quality parafac \
  --data data/processed \
  --output runs/parafac_bod \
  --targets BOD --pf-rank 5
```

Raw-EEM experiments use substantially more memory. Run them one target or one feature family at a time. The raw-EEM Linear Regression results are collected in `runs/original_eem_comparison.csv`.

## Neural models

Neural models use the EEM matrix as an image, optionally with a log FFT-magnitude channel. Available architectures are `cnn`, `resnet10` and `resnet18`; available tabular feature sets are `EEM_only`, `EC`, `SS` and `EC_SS`.

```bash
PYTHONPATH=src .venv/bin/python -m eem_water_quality neural \
  --data data/processed \
  --output runs/neural_bod \
  --targets BOD \
  --models resnet10 \
  --feature-sets EC_SS \
  --epochs 20 --patience 5 --device cpu
```

Neural training uses the validation partition for early stopping and evaluates the selected checkpoint on the test partition.

## Evaluation and results

The CLI uses a deterministic random split with `seed=42`, 608 training rows and 152 test rows for the current dataset. Classical ML currently combines the provisional train and validation rows for fitting and uses the test partition for the reported metric and model selection. Therefore the classical results below are a fixed-split benchmark; they should not be treated as an unbiased final estimate until model selection is separated from test evaluation.

The table reports test R². Each experiment has a separate LR and XGB row.
`—` means that the model has not been run or that the experiment was skipped
for TOC because it contains TOC as a feature. Original-EEM XGBoost rows are
also left open for a later memory-safe run.

| Experiment | Model | BOD | COD | TOC | BOD/COD |
|---|---|---:|---:|---:|---:|
| TOC | LR | 0.700 | 0.862 | — | 0.178 |
| TOC | XGB | 0.373 | 0.790 | — | -0.314 |
| TOC_SS_EC | LR | 0.711 | 0.868 | — | 0.299 |
| TOC_SS_EC | XGB | 0.521 | 0.841 | — | 0.072 |
| EEMpca | LR | 0.332 | 0.469 | 0.505 | 0.148 |
| EEMpca | XGB | 0.455 | 0.634 | 0.740 | 0.395 |
| EEMpca_TOC | LR | 0.737 | 0.837 | — | 0.279 |
| EEMpca_TOC | XGB | 0.707 | 0.834 | — | 0.416 |
| EEMpca_SS_EC | LR | 0.374 | 0.472 | 0.513 | 0.286 |
| EEMpca_SS_EC | XGB | 0.486 | 0.741 | 0.721 | 0.425 |
| EEMpca_TOC_SS_EC | LR | 0.747 | 0.840 | — | 0.377 |
| EEMpca_TOC_SS_EC | XGB | 0.738 | 0.842 | — | 0.364 |
| EEM | LR | 0.295 | -0.157 | 0.421 | -2.939 |
| EEM | XGB | — | — | — | — |
| EEM_TOC | LR | 0.536 | -0.208 | — | -2.806 |
| EEM_TOC | XGB | — | — | — | — |
| EEM_SS_EC | LR | -0.752 | -0.231 | 0.438 | -3.410 |
| EEM_SS_EC | XGB | — | — | — | — |
| EEM_TOC_SS_EC | LR | -0.614 | -0.216 | — | -3.405 |
| EEM_TOC_SS_EC | XGB | — | — | — | — |

The complete four-target no-leakage run is under
`runs/selected_4targets_lr_xgb_no_leakage/`. Individual raw-EEM runs and their
predictions are under `runs/original_eem_*_linear/`.

Each completed run contains `run.json`, `splits.csv`, a root `test_metrics.csv`, and one directory per target (`BOD/`, `COD/`, `TOC/`, `BOD_COD/`). Target directories contain metrics, predictions, `selected.json` and the fitted model/preprocessing artifacts.

## Checks

```bash
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/ruff check src tests scripts
```
