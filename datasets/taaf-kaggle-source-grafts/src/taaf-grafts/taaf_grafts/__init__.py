"""Graft modules for the forked duck harness (assigned in notebook cell 12).

Everything in this package must import cleanly with only the vendored
``tufa-arc-agi-framework`` / ``ARC3-Inference`` trees on ``sys.path`` — on
Kaggle that is exactly what cell 8 provides, locally ``taaf_rig/env.py``
provides the same. No module here may start GPU work at import time.
"""

from taaf_grafts.banking_solver import BankingHarnessSolver

__all__ = ["BankingHarnessSolver"]
