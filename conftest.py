"""Pytest session setup.

Pin the native BLAS / OpenMP thread pools to a single thread so the heavy LightGBM / SHAP test
modules don't oversubscribe threads and get the process killed by the sandbox's thread/process cap
when the WHOLE suite runs in one `pytest` invocation (this presented as a spurious "OOM"/exit-137
that truncated the run, never an assertion failure). Single-threaded is more than enough for the
small in-test fixtures and makes a one-shot full-suite run reliable here.

These must be set BEFORE numpy / lightgbm initialise their native pools, so this lives at the top
of the rootdir ``conftest.py`` (imported during pytest bootstrap, before any test module). We use
``setdefault`` so an explicit environment override always wins.
"""
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
