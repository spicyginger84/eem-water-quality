# EEM water-quality experiments

This project reconstructs the notebook experiments as reproducible command-line runs.
Processed inputs are read from `data/processed` (`eem.npy`, `samples.parquet`, and
`wavelengths.npz`). No feature transformer is fitted on validation or test rows.

Install the package and development dependencies with:

```bash
python -m pip install -e '.[dev]'
```

Inspect the processed data and list the available experiments:

```bash
eem-quality inspect
eem-quality list
```

Run the classical models with grouped location splits (the default):

```bash
eem-quality ml --targets BOD_COD --models linear tree --experiments EEMpca PARAFAC_TOC_SS_EC
```

Run the nonnegative PARAFAC + SVR baseline:

```bash
eem-quality parafac --targets BOD --pf-rank 5
```

The optional neural command requires the `neural` extra:

```bash
python -m pip install -e '.[neural]'
eem-quality neural --targets BOD --models resnet10 --feature-sets EC_SS
```

Each run writes its split assignments, configuration, data hashes, held-out test
metrics, predictions, and fitted model artifacts under `runs/`. Classical ML
combines the provisional train and validation rows and evaluates every model on
the held-out test partition, matching the original notebook protocol. Neural
experiments keep a validation partition for early stopping and model selection.

Run checks with `pytest -q` and `ruff check src tests`.
