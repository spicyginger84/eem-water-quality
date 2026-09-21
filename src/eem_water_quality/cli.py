"""Command-line entry points; neural dependencies are loaded only when requested."""

import argparse
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
    for name in ["ml", "parafac", "neural"]:
        command = sub.add_parser(name)
        command.add_argument("--data", default="data/processed")
        command.add_argument("--output", default=None)
        command.add_argument(
            "--targets", nargs="+", default=["BOD" if name == "neural" else "BOD_COD"]
        )
        command.add_argument("--split", choices=["random"], default="random")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--test-size", type=float, default=0.2)
        command.add_argument(
            "--val-size",
            type=float,
            default=0.2,
            help="Validation fraction of the non-test rows",
        )
        command.add_argument("--n-jobs", type=positive_int, default=1)
        if name == "neural":
            command.add_argument(
                "--models", nargs="+", choices=["cnn", "resnet10", "resnet18"], default=["resnet10"]
            )
            command.add_argument(
                "--feature-sets",
                nargs="+",
                choices=["EEM_only", "EC", "SS", "EC_SS"],
                default=["EC_SS"],
            )
            command.add_argument("--fft", action=argparse.BooleanOptionalAction, default=True)
            command.add_argument("--epochs", type=positive_int, default=200)
            command.add_argument("--patience", type=positive_int, default=20)
            command.add_argument("--batch-size", type=positive_int, default=32)
            command.add_argument("--lr", type=float, default=1e-3)
            command.add_argument("--weight-decay", type=float, default=1e-4)
            command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        else:
            command.add_argument(
                "--models",
                nargs="+",
                choices=["linear", "tree", "xgboost", "svr"],
                default=["linear", "tree", "xgboost"],
            )
            command.add_argument("--experiments", nargs="+", choices=list(experiment_catalog()))
            command.add_argument("--pca-components", type=positive_int, default=30)
            command.add_argument(
                "--pf-rank", type=positive_int, default=5 if name == "parafac" else 4
            )
            command.add_argument(
                "--pf-max-iter", type=positive_int, default=500 if name == "parafac" else 1000
            )
            command.add_argument("--log-targets", nargs="*", default=[])
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == "list":
        for name, exp in experiment_catalog().items():
            print(f"{name}: EEM={exp.eem}, PARAFAC={exp.parafac}, tabular={exp.tabular}")
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
    if args.command == "neural":
        from .neural import run_neural

        output = run_neural(args)
    else:
        from .ml import run_ml

        output = run_ml(args)
    print(f"Saved run to {output}")
