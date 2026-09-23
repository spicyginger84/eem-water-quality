"""Run provenance helpers.

The current artifact writer stores configuration, package versions, git state
and input hashes in ``run.json``.  This module is the named home for that
responsibility while the low-level CSV/model writer remains in ``io.py``.
"""

from .io import start_run, write_json

__all__ = ["start_run", "write_json"]

