"""Compatibility entry point for the correlation diagnostic."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parents[1] / "plot_correlations.py"), run_name="__main__")

