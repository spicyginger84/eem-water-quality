"""Processed dataset contract, I/O and split utilities."""

from .io import load_data, load_processed_dataset
from .schema import (
    ALIASES,
    BOD,
    COD,
    EC,
    PH,
    SS,
    TEMP,
    TOC,
    ProcessedDatasetSchema,
    resolve_column,
)
from .splitting import group_kfold_indices, split_indices, target_partitions

__all__ = [
    "ALIASES",
    "BOD",
    "COD",
    "EC",
    "PH",
    "SS",
    "TEMP",
    "TOC",
    "ProcessedDatasetSchema",
    "group_kfold_indices",
    "load_data",
    "load_processed_dataset",
    "resolve_column",
    "split_indices",
    "target_partitions",
]

