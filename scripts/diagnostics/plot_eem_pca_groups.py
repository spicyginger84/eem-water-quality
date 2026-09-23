"""Compatibility entry point for the EEM PCA group diagnostic."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).parents[1] / "plot_eem_pca_groups.py"), run_name="__main__"
    )

