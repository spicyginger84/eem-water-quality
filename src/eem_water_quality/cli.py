"""Command-line entry points for the classical benchmark pipeline."""

import argparse
import logging
from datetime import UTC, datetime

from .data import load_data
from .features import experiment_catalog


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def make_parser():
    parser = argparse.ArgumentParser(description="EEM water-quality regression experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate and summarize processed data")
    inspect.add_argument("--data", default="data/processed")
    sub.add_parser("list", help="List all reconstructed classical feature experiments")
    for name in ["ml"]:
        command = sub.add_parser(name)
        command.add_argument("--data", default="data/processed")
        command.add_argument("--output", default=None)
        command.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="verbosity of progress logs (default: INFO)",
        )
        command.add_argument(
            "--targets",
            nargs="+",
            default=["BOD", "COD", "TOC", "BOD_COD"],
            help="targets to evaluate (default: BOD COD TOC BOD_COD)",
        )
        command.add_argument(
            "--split",
            choices=["random", "group", "two_way"],
            default="group",
            help="random, station-group, or station/month two-way split (default: group)",
        )
        command.add_argument(
            "--group-col",
            default="Point",
            help="column used for group-disjoint splitting",
        )
        command.add_argument(
            "--secondary-group-col",
            default="Month",
            help="second grouping column for --split two_way (default: Month)",
        )
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--test-size", type=float, default=0.2)
        command.add_argument(
            "--val-size",
            type=float,
            default=0.2,
            help="Validation fraction of the non-test rows",
        )
        command.add_argument("--n-jobs", type=positive_int, default=1)
        feature_choices = list(experiment_catalog())
        command.add_argument(
            "--models",
            nargs="+",
            choices=["linear", "tree", "xgboost", "svr"],
            default=["linear", "svr", "tree", "xgboost"],
            help="models to evaluate (default: linear svr tree xgboost)",
        )
        command.add_argument(
            "--features",
            nargs="+",
            choices=feature_choices,
            default=["EEMpca"],
            help="feature representations to benchmark (default: EEMpca)",
        )
        # Backward-compatible alias. New commands and documentation use
        # --features; keeping this hidden avoids breaking saved commands.
        command.add_argument(
            "--experiments",
            dest="features",
            nargs="+",
            choices=feature_choices,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--evaluation-protocol",
            choices=["cv", "single_split"],
            default="cv",
            help="cross-validation by default; single_split is compatibility mode",
        )
        command.add_argument(
            "--cv-folds",
            type=positive_int,
            default=5,
            help="number of folds for random/group CV (default: 5)",
        )
        command.add_argument(
            "--holdout-protocol",
            choices=["loso", "lomo"],
            default=None,
            help="override CV with leave-one-station/month-out",
        )
        command.add_argument("--pca-components", type=positive_int, default=30)
        command.add_argument("--log-targets", nargs="*", default=[])
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", "INFO")),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.command == "list":
        for name, exp in experiment_catalog().items():
            print(f"{name}: EEM={exp.eem}, tabular={exp.tabular}")
        return
    if args.command == "inspect":
        eem, samples, wavelengths = load_data(args.data)
        print(f"EEM shape: {eem.shape}; dtype: {eem.dtype}; finite: True")
        print(f"Metadata: {samples.shape}; wavelength arrays: {list(wavelengths)}")
        print(f"Columns: {samples.columns.tolist()}")
        print(samples.describe().to_string())
        return
    if args.output is None:
        args.output = (
            "runs/" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "_" + args.command
        )
    from .pipelines.classical import run_ml

    output = run_ml(args)
    print(f"Saved run to {output}")
