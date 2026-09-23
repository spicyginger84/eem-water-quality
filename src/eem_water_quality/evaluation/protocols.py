"""Resolve the mutually exclusive evaluation protocols."""

from __future__ import annotations

import numpy as np

from ..data import split_indices
from .cross_validation import grouped_cv_splits, leave_one_group_out_splits, random_cv_splits


def resolve_protocol(args) -> str:
    """Resolve the effective protocol and reject incompatible options."""
    holdout = getattr(args, "holdout_protocol", None)
    split = getattr(args, "split", "group")
    evaluation = getattr(args, "evaluation_protocol", "cv")
    if holdout:
        if split == "two_way":
            raise ValueError("--holdout-protocol cannot be combined with --split two_way")
        return holdout
    if split == "two_way":
        return "two_way"
    if evaluation not in {"cv", "single_split"}:
        raise ValueError(f"Unknown evaluation protocol: {evaluation}")
    return evaluation


def build_evaluation_splits(samples, args):
    """Build split records according to the documented precedence rules."""
    protocol = resolve_protocol(args)
    group_col = getattr(args, "group_col", "Point")
    if protocol == "cv":
        if args.split == "random":
            return protocol, random_cv_splits(len(samples), args.cv_folds, args.seed)
        if args.split == "group":
            return protocol, grouped_cv_splits(samples, group_col, args.cv_folds, args.seed)
        raise ValueError("Cross-validation supports --split random or --split group")
    if protocol == "loso":
        return protocol, leave_one_group_out_splits(samples, group_col)
    if protocol == "lomo":
        configured_group = getattr(args, "group_col", "Point")
        month_col = "Month" if configured_group == "Point" else configured_group
        return protocol, leave_one_group_out_splits(samples, month_col)
    if protocol == "two_way":
        split = split_indices(
            samples,
            mode="two_way",
            group_col=group_col,
            seed=args.seed,
            test_size=args.test_size,
            val_size=args.val_size,
            secondary_group_col=getattr(args, "secondary_group_col", "Month"),
        )
        train = np.concatenate([split["train"], split["validation"]])
        return protocol, [
            {
                "fold": 1,
                "train": train,
                "test": split["test"],
                "excluded": split.get("excluded", np.array([], dtype=int)),
                "heldout_group": "two_way",
            }
        ]
    if protocol == "single_split":
        split = split_indices(
            samples,
            mode=args.split,
            group_col=group_col,
            seed=args.seed,
            test_size=args.test_size,
            val_size=args.val_size,
            secondary_group_col=getattr(args, "secondary_group_col", "Month"),
        )
        return protocol, [
            {
                "fold": 1,
                "train": split["train"],
                "validation": split["validation"],
                "test": split["test"],
                "excluded": split.get("excluded", np.array([], dtype=int)),
            }
        ]
    raise ValueError(f"Unsupported protocol: {protocol}")
