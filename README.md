# EEM Water-Quality: GMM and PARAFAC

Research workflows for decomposing excitation–emission matrices (EEMs) and
estimating BOD, COD and TOC. The active comparison uses improved V0 (called GMM
in the report) and PARAFAC.

- [Model, assumptions and results](docs/gmm_improved_v0_report.md)
- [Station grouping and membership](docs/station_grouping_analysis.md)
- [Sampling calendar](docs/sampling_calendar_report.md)

## Setup

```bash
python -m pip install -e ".[notebooks,boosting,dev]"
```

Select this Python environment as the notebook kernel. Processed data must be
available under `data/processed/`.

## Active workflows

| Entry point | Purpose | Output |
|---|---|---|
| [improved_v0_10stations.ipynb](notebooks/improved_v0_10stations.ipynb) | GMM by group, temporal evaluation and component regression; filename retains the historical station name | `runs/diagnostics/v0_improved_groups/` |
| [parafac.ipynb](notebooks/parafac.ipynb) | PARAFAC by group, rank diagnostics and component regression | `runs/diagnostics/parafac/by_group/` |
| [run_improved_v0.py](scripts/diagnostics/run_improved_v0.py) | Reproduce the historical ten-station GMM experiment | `runs/diagnostics/v0_improved_10stations/` |
| [evaluate_v0_component_regression.py](scripts/diagnostics/evaluate_v0_component_regression.py) | Single/all-component GMM regression from saved models | `<experiment>/component_regression/` |
| [evaluate_parafac_all_components.py](scripts/diagnostics/evaluate_parafac_all_components.py) | PARAFAC regression using all components | `runs/diagnostics/parafac/by_group/` |
| [compare_improved_v0.py](scripts/diagnostics/compare_improved_v0.py) | Compare saved ten-station results, including historical V0 | `runs/diagnostics/model_comparison/improved_v0_vs_v0_parafac/` |
| [fit_gayang_parafac.py](scripts/diagnostics/fit_gayang_parafac.py) | Standalone masked PARAFAC diagnostic solver | Output selected by `--output` |

Run notebooks from top to bottom. Fitting can be expensive; the GMM notebook
uses cached fits. Regression-only CV on full-data decomposition is distinct
from temporal evaluation that refits shapes within each training fold.

The report contains results for the six largest groups and a separate historical
ten-station benchmark. These are overlapping data, not independent test sets.

## Data

There are 760 aligned samples from 45 stations. Each EEM has shape
`(271 emission, 57 excitation)`.

```text
data/processed/
  eem.npy                  # EEMs in sample-table order
  samples.csv              # metadata, Point, Group and laboratory targets
  samples.parquet          # equivalent metadata table
  wavelengths.npz          # excitation and emission axes
  station_groups.csv       # fixed Point-to-Group mapping
  eem_mapping.csv          # raw-file provenance
  locations.csv            # station metadata; also available as Parquet
  processing_summary.json  # processing counts
```

Group labels are applied without reordering samples:

```bash
.venv/bin/python scripts/data/assign_station_groups.py
```

Keep `eem.npy` and the sample tables aligned. `data/` and `LOCAL/` retain source
material and existing data archives. The repository currently has no working
raw-workbook processing entry point; the removed compatibility wrapper referred
to a missing implementation. Restoring an old processed archive may require
reapplying the group mapping.

## Code and retained artifacts

- `src/eem_water_quality/features/improved_v0.py`: Gaussian shapes and fitting.
- `src/eem_water_quality/features/peak_decomposition.py`: NNLS, masks and shared
  decomposition utilities.
- `src/eem_water_quality/`: shared data, model and evaluation infrastructure;
  the classical CLI remains available, but its old run artifacts were removed.
- `tests/`: numerical and pipeline checks.
- `docs/assets/`: grouping/calendar inputs and cleanup inventory.

Keep the following output directories to reproduce the current report:

```text
runs/diagnostics/
  v0_improved_groups/
  v0_improved_10stations/
  parafac/
  model_comparison/
  shared_gaussian_v0/by_station_top10_unmasked/
```

The last directory remains because the ten-station runner and comparison script
read historical V0 artifacts. The old V0 notebook and standalone theory report
were removed; retaining its artifacts does not make it an active workflow.

The deletion inventory is in
[repository_cleanup.json](docs/assets/repository_cleanup.json). Raw data,
current report, active models and their result tables were retained.

## Checks

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/pytest -q
```
