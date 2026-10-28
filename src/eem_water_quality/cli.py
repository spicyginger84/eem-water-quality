"""Command line: `eem-quality inspect` and `eem-quality run CONFIG --out DIR`."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="eem-quality", description="EEM water-quality benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Summarise the processed dataset")
    inspect.add_argument("--data", default="data/processed")
    run = sub.add_parser("run", help="Run a benchmark config")
    run.add_argument("config", nargs="?", help="JSON config (configs/benchmark/*.json); defaults if omitted")
    run.add_argument("--out", required=True, help="Run directory")
    run.add_argument("--models", help="Comma-separated override of the config's models")
    run.add_argument("--schemes", help="Comma-separated override of the config's schemes")
    run.add_argument("--workers", type=int)
    args = parser.parse_args(argv)

    if args.command == "inspect":
        import numpy as np

        from .data import load_dataset

        D = load_dataset(args.data)
        print(f"samples {len(D)}, stations {len(np.unique(D.station))}, EEM {D.x.shape[1:]} (Em step 2), "
              f"valid cells {int(D.mask.sum())}")
        print(f"targets {D.targets}: labelled {np.isfinite(D.y).sum(0).tolist()}")
        print(f"fold schemes: {sorted(D.folds)} + random, loso; brackish samples {int(D.brackish.sum())}")
        return

    from .runner import Benchmark, Config

    split = lambda s: s.split(",") if s else None
    cfg = Config.load(args.config, models=split(args.models), schemes=split(args.schemes), workers=args.workers)
    Benchmark(cfg).run(args.out)
    print(f"report: {args.out}/report.md")


if __name__ == "__main__":
    main()
