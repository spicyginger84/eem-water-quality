# EEM Water-Quality Prediction

This project predicts water-quality parameters from excitation–emission matrix (EEM) fluorescence measurements and optional laboratory covariates. Targets include biochemical oxygen demand (BOD), chemical oxygen demand (COD), total organic carbon (TOC), and the derived BOD/COD ratio.

The pipeline saves data splits, fitted preprocessing, model parameters, predictions, and metrics for every run.

## Installation

```bash
conda create -n water-quality --override-channels -c conda-forge python=3.12 -y
conda activate water-quality
python -m pip install -e '.[dev,neural,boosting]'
```

Processed inputs are expected under `data/processed`:

```text
eem.npy
samples.parquet
wavelengths.npz
```

Inspect the data and list available experiments:

```bash
eem-quality inspect --data data/processed
eem-quality list
```

## Classical ML experiments

Feature experiments combine tabular covariates (TOC, suspended solids, and electrical conductivity), raw or PCA-reduced EEM features, and PARAFAC component scores. Available models are linear regression, decision tree, SVR, and optional XGBoost.

```bash
eem-quality ml --data data/processed --output runs/ml_bod \
  --targets BOD --split random --models linear xgboost \
  --experiments EEMpca_PARAFAC_SS_EC
```

```bash
eem-quality ml --data data/processed --output runs/ml_bod_cod_toc \
  --targets BOD COD TOC --split random --models linear xgboost \
  --experiments TOC TOC_SS_EC PARAFAC PARAFAC_SS_EC \
  EEMpca EEMpca_TOC EEMpca_SS_EC 
```

Use grouped location splits for unseen sampling points:

```bash
eem-quality ml --data data/processed --output runs/ml_grouped \
  --targets BOD_COD --split group --group-col Point \
  --models linear xgboost
```

Run the standalone nonnegative PARAFAC + SVR model:

```bash
eem-quality parafac --data data/processed --output runs/parafac_bod \
  --targets BOD --pf-rank 5
```

## Deep-learning experiments

Neural models use EEM images with an optional FFT-magnitude channel and can fuse tabular features. Available architectures are `cnn`, `resnet10`, and `resnet18`.

```bash
eem-quality neural --data data/processed --output runs/neural_bod \
  --targets BOD --models resnet10 --feature-sets EC_SS
```

For a short CPU smoke run:

```bash
eem-quality neural --data data/processed --output runs/neural_quick \
  --targets BOD --models resnet10 --feature-sets EC_SS \
  --epochs 20 --patience 5 --device cpu
```

## Saved results

Each run contains:

```text
run.json                         configuration, versions, and data hashes
splits.csv                       sample-to-partition assignments
test_metrics.csv                 test metrics for evaluated configurations
target_00/                       artifacts for the first target
├── validation_metrics.csv       neural validation metrics, when applicable
├── test_predictions.csv         predictions for the selected configuration
├── selected.json                selected model and final metrics
├── model.joblib or model.pt     fitted model
└── preprocessing.joblib         fitted preprocessing objects
```

Metrics include R², MSE, RMSE, MAE, and MAPE. Run checks with:

```bash
pytest -q
ruff check src tests
```
