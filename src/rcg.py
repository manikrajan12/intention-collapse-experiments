"""Compatibility wrapper for importing RCG helpers from notebook-style src imports."""

from pathlib import Path
import importlib.util


_RCG_PATH = Path(__file__).resolve().parent.parent / "metrics" / "rcg.py"
_SPEC = importlib.util.spec_from_file_location("intention_collapse_metrics_rcg", _RCG_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load RCG module from {_RCG_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

sample_hidden_states = _MODULE.sample_hidden_states
compute_recoverability_scores = _MODULE.compute_recoverability_scores
compute_rcg = _MODULE.compute_rcg
compute_rcg_full = _MODULE.compute_rcg_full
plot_rcg_trajectory = _MODULE.plot_rcg_trajectory

__all__ = [
    "sample_hidden_states",
    "compute_recoverability_scores",
    "compute_rcg",
    "compute_rcg_full",
    "plot_rcg_trajectory",
]
