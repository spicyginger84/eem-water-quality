# EEM Water-Quality Prediction

This repository benchmarks classical regression models for predicting BOD, COD,
TOC and the derived BOD/COD ratio from excitation–emission matrices (EEMs).
The current benchmark includes Linear Regression, SVR, Decision Tree and
XGBoost. PARAFAC and deep-learning models are outside the current pipeline.

The main entry point is:

```bash
PYTHONPATH=src python -m eem_water_quality ml ...
```

The current evaluation is deliberately explicit about station and month
generalization. The Phase 2 results are documented in
[docs/phase2_report.md](docs/phase2_report.md), and the research checklist is
in [docs/Improvement_plan.md](docs/Improvement_plan.md).

## Environment

Install the package with the optional XGBoost and development dependencies when
running the complete benchmark:

```bash
python -m pip install -e ".[dev,boosting]"
```

If XGBoost is not installed, run the same commands with `--models linear svr
tree`.

## Repository layout

```text
src/eem_water_quality/
├── __main__.py             # python -m entry point
├── cli.py                  # inspect, list and ml commands
├── data/                   # processed-data loader, schema and split helpers
├── preprocessing/          # EEM and tabular preprocessing helpers
├── features/               # raw EEM, PCA and tabular feature catalog
├── models/                 # classical model factories
├── evaluation/             # protocols, CV splitters, baselines and metrics
├── pipelines/              # benchmark orchestration
├── artifacts/              # run manifests and output writers
└── predict.py              # reload a saved classical fold artifact

scripts/
└── diagnostics/            # target/station and EEM-distance diagnostics
```

The benchmark consumes the processed-data contract below. Notebooks and raw
workbooks are not required to run the ML pipeline.

## Processed data

The current processed dataset contains 760 aligned samples from 45 stations.
Each EEM has shape `(271 emission, 57 excitation)` on the common wavelength
grid, giving 15,447 flattened cells. The sample table contains station (`Point`),
month (`Month`), round, depth, sampling metadata and laboratory parameters.

The required files are:

```text
data/processed/
├── eem.npy                  # float32 array, shape (n_samples, 271, 57)
├── samples.parquet          # aligned metadata and target columns
├── samples.csv              # CSV copy of the sample table
├── wavelengths.npz          # emission and excitation axes
├── eem_mapping.csv          # mapping/provenance for raw EEM files
├── locations.parquet        # station metadata
├── locations.csv            # CSV copy of station metadata
└── processing_summary.json  # processing filters and counts
```

If the processed archive is used, restore it with:

```bash
unzip data/processed.zip -d data
```

The mapping metadata records the station, month, round, source filename,
laboratory row and matching method. The processed data currently records 760
matched samples, 239 unmatched raw files and 2 samples removed by the depth
limit. The default processing rule keeps samples with `Depth <= 2 m` and
requires numeric BOD, COD, TOC, SS and EC values. Missing depth is retained.

Inspect the processed contract and list all registered feature sets with:

```bash
PYTHONPATH=src python -m eem_water_quality inspect --data data/processed
PYTHONPATH=src python -m eem_water_quality list
```

## Feature construction

Feature transformations are fit on training rows inside each fold and then
applied to the held-out rows:

- `EEM`: flattened processed EEM values.
- `EEMpca`: training-only standardization followed by PCA; the default rank is
  30. Cells that are zero in every training sample are removed before scaling
  and PCA.
- `SS_EC`, `Temp`, `pH` and their registered combinations: numeric tabular
  covariates with median imputation and scaling.
- `EEM_*` and `EEMpca_*`: EEM features concatenated with selected tabular
  covariates.

Use `python -m eem_water_quality list` for the complete catalog. The CLI uses
the option name `--features`; the hidden `--experiments` alias is retained only
for old commands.

TOC-containing feature sets are skipped when the target is TOC, preventing
laboratory-target leakage. The derived `BOD_COD` target is calculated when the
processed table is loaded and is never used as an input feature.

## Models

The available model names are:

| CLI name | Estimator |
|---|---|
| `linear` | Linear Regression |
| `svr` | standardized RBF Support Vector Regression |
| `tree` | Decision Tree Regression |
| `xgboost` | XGBoost Regression |

If `--targets` is omitted, the CLI evaluates all four targets:
`BOD`, `COD`, `TOC` and `BOD_COD`.
If `--models` is omitted, it evaluates all four models: `linear`, `svr`,
`tree` and `xgboost`.

## Benchmark protocols

The default is grouped five-fold cross-validation:

```bash
PYTHONPATH=src python -m eem_water_quality ml \
  --data data/processed \
  --output runs/grouped5fold_eempca \
  --targets BOD COD TOC BOD_COD \
  --features EEMpca EEMpca_SS_EC \
  --models linear svr tree xgboost \
  --split group --group-col Point \
  --cv-folds 5 --pca-components 30 --n-jobs 1
```

`--split group` uses GroupKFold by station. `--split random` uses sample-level
KFold. Every requested target, feature set and model uses the same folds; no
model is selected from a held-out fold. The summaries contain arithmetic mean,
standard deviation and pooled R². Global-mean and station-mean baselines are
also fit from each training fold. For an unseen station, the station baseline
falls back to the global mean to avoid target leakage.

The pipeline logs target, fold, sample counts, feature timing and model timing
at `INFO` level. Add `--log-level DEBUG` for more detailed progress output.

### Leave-one-station-out (LOSO)

```bash
PYTHONPATH=src python -m eem_water_quality ml \
  --data data/processed \
  --output runs/loso_eempca \
  --targets BOD COD TOC BOD_COD \
  --features EEMpca \
  --models linear svr tree xgboost \
  --holdout-protocol loso --group-col Point \
  --pca-components 30 --n-jobs 1
```

Each fold holds out one station entirely. There is no train/test ratio and no
additional five-fold CV inside LOSO.

### Leave-one-month-out (LOMO)

```bash
PYTHONPATH=src python -m eem_water_quality ml \
  --data data/processed \
  --output runs/lomo_eempca \
  --targets BOD COD TOC BOD_COD \
  --features EEMpca \
  --models linear svr tree xgboost \
  --holdout-protocol lomo --group-col Month \
  --pca-components 30 --n-jobs 1
```

Each fold holds out one month entirely. Stations can still appear in the
training months, so the station-mean baseline is meaningful in this protocol.

### Two-way holdout

```bash
PYTHONPATH=src python -m eem_water_quality ml \
  --data data/processed \
  --output runs/two_way_eempca \
  --targets BOD COD TOC BOD_COD \
  --features EEMpca \
  --models linear svr tree xgboost \
  --split two_way --group-col Point --secondary-group-col Month \
  --test-size 0.2 --val-size 0.2 --seed 42 \
  --pca-components 30 --n-jobs 1
```

This is one strict holdout where the test samples belong simultaneously to an
unseen station and an unseen month. Samples that belong to only one held-out
group are excluded. The internal validation portion is merged into training
before the final fit; two-way does not run cross-validation.

For compatibility, `--evaluation-protocol single_split` creates one grouped,
random or two-way train/validation/test split and writes separate validation
and test files. It is not the default benchmark and does not perform model
selection.

## Run artifacts

If `--output` is omitted, the CLI creates a timestamped directory under
`runs/`. A completed run contains:

```text
runs/<run>/
├── run.json                 # configuration, versions and input hashes
├── splits.csv               # row-level split manifest
├── fold_metrics.csv        # all fold-level rows
├── summary_metrics.csv     # aggregated rows
└── <target>/
    ├── target.json
    ├── cv_fold_metrics.csv / holdout_metrics.csv
    ├── cv_summary.csv      / holdout_summary.csv
    ├── cv_predictions.csv  / holdout_predictions.csv
    └── ...
```

Each requested feature/model/fold also has a serialized classical model under
`models/<target>/<features>/<model>/fold_<id>/`. Metrics include R², MSE, RMSE,
MAE and MAPE. No result file contains a model chosen using the held-out test
rows.

## Diagnostics

Summarize target ranges by station and identify LOSO target extrapolation:

```bash
PYTHONPATH=src python scripts/diagnostics/analyze_target_by_station.py \
  --data data/processed \
  --output runs/diagnostics/target_by_station \
  --targets BOD COD TOC BOD_COD
```

Measure fold-local EEM distance from each held-out station to training stations
and correlate it with the saved LOSO R²:

```bash
PYTHONPATH=src python scripts/diagnostics/analyze_loso_eem_distance.py \
  --data data/processed \
  --run runs/loso_eempca \
  --output runs/diagnostics/loso_eem_distance \
  --targets BOD COD TOC BOD_COD \
  --features EEMpca \
  --models linear svr tree xgboost \
  --distance-space pca --components 30
```

The Phase 2 report combines these diagnostics with the LOSO, LOMO and two-way
results: [docs/phase2_report.md](docs/phase2_report.md).

## Development checks

```bash
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/ruff check src tests scripts
```
