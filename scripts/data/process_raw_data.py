"""Compatibility entry point for the raw-data processor.

The implementation remains in the historical top-level script while the
repository is migrated to the documented scripts/data namespace.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parents[1] / "process_raw_data.py"), run_name="__main__")

