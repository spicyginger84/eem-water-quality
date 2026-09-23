# EEM Water-Quality Prediction

This project predicts laboratory water-quality parameters from excitation–emission matrix (EEM) fluorescence measurements. The main targets are BOD, COD, TOC and the derived BOD/COD ratio. Models can use EEM features alone, EEM features combined with laboratory covariates, or tabular covariates alone.

Repository structure, naming conventions, pipeline contracts and artifact schemas
are documented in [docs/repository_architecture.md](docs/repository_architecture.md).

## Data

The raw data is under `data/2026ER_data_for Viet`:

- Monthly folders contain one EEM workbook per sample.
- `20260120_ER_data_HS.xlsx`, sheet `Data`, contains laboratory parameters with `Label`, `Month` and `Round`.
- The `Location` sheet contains station metadata.
- The workbook records sampling year 2023 even though the directory name contains `2026ER`.

The processed dataset currently contains 760 aligned samples from 45 stations. Each EEM has 271 emission values from 280–550 nm and 57 excitation values from 220–500 nm, so a flattened EEM contains 15,447 features. The matrix keeps this order as `(sample, emission, excitation)`.

The processed files are:

```text
data/processed/
├── eem.npy                  EEM array with shape (760, 271, 57)
├── samples.parquet          aligned metadata and laboratory parameters
├── samples.csv              CSV copy of samples.parquet
├── wavelengths.npz          emission and excitation wavelengths
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

If the raw archive is extracted with its leading `data/` directory, the input
root may be `data/data/2026ER_data_for Viet`; pass that path to `--raw`.

To apply the optional physical consistency check, add
`--reject-bod-greater-cod`; those rows are recorded as `quality_filtered` in
`eem_mapping.csv` rather than silently removed.

The mapper uses `Label + Month + Round` as the sample key. A filename with a round, such as `ha1-1_3.xlsx`, is interpreted as station `Ha1`, round `1`, in month `3`. A filename containing only a station is accepted only when exactly one laboratory row exists for that station and month. Ambiguous files such as `ha20_3.xlsx` are recorded as `unmatched` instead of being assigned to an arbitrary round.

Each EEM workbook is checked for a numeric wavelength header, a consistent matrix shape, finite values and the common wavelength grid. The default completeness filter requires numeric BOD, COD, TOC, SS and EC, treats `ND` and blanks as missing, and excludes samples with `Depth > 2 m`. Missing depth is retained. A mapping row is only accepted when the station, month and round identify one laboratory row, so unmatched and ambiguous filenames remain auditable.

The current mapping contains 760 accepted samples, 239 files without a unique laboratory match and 2 files excluded by the depth limit. `eem_mapping.csv` records the source-relative filename, laboratory row, matching method and exclusion reason, so the mapping remains portable when the raw root is moved.

Inspect the processed data and list the available feature experiments:

```bash
PYTHONPATH=src .venv/bin/python -m eem_water_quality inspect --data data/processed
PYTHONPATH=src .venv/bin/python -m eem_water_quality list
```

## Feature representations

The classical pipeline builds features using the training partition only:

- `TOC`, `SS`, `EC`, `Temp` and `pH`: numeric tabular covariates with median imputation and standard scaling.
- `EEM`: flattened original EEM values (15,447 features).
- `EEMpca`: standardized EEM followed by PCA; the default is 30 components. Cells that are zero in every training sample (the fixed Rayleigh/scatter mask) are removed before scaling and PCA.
- `PARAFAC`: rank-4 PARAFAC scores by default. The standalone `parafac` command uses a nonnegative PARAFAC model.
- Combined experiments concatenate EEM/PCA/PARAFAC features with selected tabular covariates.

The target column is never used as a predictor. In particular, every
experiment that declares `TOC` as a covariate is skipped when the target is
TOC, including `TOC_SS_EC`, `EEMpca_TOC` and `EEMpca_TOC_SS_EC`. The skipped
experiment is recorded in that target's `validation_metrics.csv`.

The experiment catalog includes `TOC`, `SS_EC`, `TOC_SS_EC`, `Temp`, `pH` and their combinations with `SS_EC`, plus all `EEM`, `EEMpca` and `PARAFAC` variants with these tabular covariates.

## Classical models

Available models are:

- `linear`: Linear Regression
- `tree`: Decision Tree Regression
- `svr`: standardized RBF Support Vector Regression
- `xgboost`: XGBoost Regression

Example using PCA EEM and non-target tabular covariates:

```bash
PYTHONPATH=src .venv/bin/python -m eem_water_quality ml \
  --data data/processed \
  --output runs/ml_experiment \
  --targets BOD COD TOC \
  --split group --group-col Point --seed 42 \
  --models linear xgboost tree svr \
  --experiments EEMpca EEMpca_SS_EC \
  --pca-components 30 --n-jobs 1
```

Experiments containing TOC are intentionally excluded when evaluating TOC itself,
because TOC is a laboratory covariate rather than an EEM feature.

The standalone nonnegative PARAFAC + SVR command is:

```bash
PYTHONPATH=src .venv/bin/python -m eem_water_quality parafac \
  --data data/processed \
  --output runs/parafac_experiment \
  --targets BOD --pf-rank 5
```

Raw-EEM experiments use substantially more memory and are not included in the
current grouped benchmark. Run them one target or one feature family at a time.

## Neural models

Neural models use the EEM matrix as an image, optionally with a log FFT-magnitude channel. Available architectures are `cnn`, `resnet10` and `resnet18`; available tabular feature sets are `EEM_only`, `EC`, `SS`, `EC_SS`, `Temp`, `pH`, `Temp_pH`, `Temp_SS_EC`, `pH_SS_EC` and `Temp_pH_SS_EC`.

```bash
PYTHONPATH=src .venv/bin/python -m eem_water_quality neural \
  --data data/processed \
  --output runs/neural_experiment \
  --targets BOD \
  --models resnet10 \
  --feature-sets EC_SS \
  --epochs 20 --patience 5 --device cpu
```

Neural training uses the validation partition for early stopping and evaluates the selected checkpoint on the test partition.

## Evaluation protocol

The classical `ml` and `parafac` commands use a deterministic three-way split. With
`--split group`, station or month groups are disjoint across train, validation and
test. The `two_way` mode holds out the intersection of an unseen primary group and
an unseen secondary group; rows in the other intersections are excluded.

The main training path does **not** run 5-fold cross-validation. Each experiment/model
is fitted on the training partition and scored on the independent validation partition.
The candidate with the lowest validation RMSE is selected, refit on train plus
validation, and evaluated once on the untouched test partition. Test rows are never
used for model selection.

Every candidate test score is retained in `test_metrics.csv`; the selected candidate
also has `test_predictions.csv` and `model.joblib`. Validation metrics and predictions
are saved separately. Global-mean and station-mean baselines are fitted from the
corresponding training rows and included for comparison. The current benchmark result
tables are intentionally not recorded in this README while the protocol is being
finalized.

Example grouped run:

```bash
PYTHONPATH=src .venv/bin/python -m eem_water_quality ml \
  --data data/processed --output runs/ml_grouped \
  --targets BOD COD TOC BOD_COD \
  --split group --group-col Point --seed 42 \
  --models linear xgboost tree svr \
  --experiments EEMpca EEMpca_SS_EC --pca-components 30 --n-jobs 1
```

Use `--split random` only when the intended task is interpolation among stations
already present in the training data. Use `--split two_way --group-col Point
--secondary-group-col Month` for the strict unseen-station/unseen-month diagnostic.

For leave-one-station-out or leave-one-month-out diagnostics, use the comparison
script. It evaluates every requested model and experiment without selecting a winner
from the held-out groups:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_holdouts.py \
  --data data/processed --output runs/holdout_by_station \
  --group-col Point --targets BOD COD TOC BOD_COD \
  --experiments EEMpca EEMpca_SS_EC --models linear xgboost tree svr --n-jobs 1

PYTHONPATH=src .venv/bin/python scripts/evaluate_holdouts.py \
  --data data/processed --output runs/holdout_by_month \
  --group-col Month --targets BOD COD TOC BOD_COD \
  --experiments EEMpca EEMpca_SS_EC --models linear xgboost tree svr --n-jobs 1
```

The holdout script writes `holdout_metrics.csv`, `holdout_predictions.csv`,
`pca_mask_summary.csv` and the split artifact. The outputs contain both arithmetic
mean and pooled R² where multiple held-out groups are evaluated.

## Checks

```bash
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/ruff check src tests scripts
```
