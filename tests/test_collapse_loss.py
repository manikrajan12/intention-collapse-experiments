"""
Tests for the derived collapse_loss metric.

collapse_loss is defined as:

    recoverability_auroc - accuracy

where accuracy is normalized to [0, 1] before subtraction.
"""

import json
import sys
from pathlib import Path
import math

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.derived_metrics import (
    compute_collapse_loss,
    add_collapse_loss_to_condition_summary,
)


def test_compute_collapse_loss_fraction_accuracy():
    """Fractional accuracy should be used directly."""
    collapse_loss = compute_collapse_loss(0.47, 0.61)
    assert math.isclose(collapse_loss, 0.14, rel_tol=0.0, abs_tol=1e-9)


def test_compute_collapse_loss_percentage_accuracy():
    """Percentage accuracy should be normalized before subtraction."""
    collapse_loss = compute_collapse_loss(47.0, 0.61)
    assert math.isclose(collapse_loss, 0.14, rel_tol=0.0, abs_tol=1e-9)


def test_compute_collapse_loss_missing_recoverability():
    """Missing recoverability should produce a graceful None."""
    assert compute_collapse_loss(0.47, None) is None


def test_condition_summary_serialization_includes_collapse_loss():
    """Condition summaries should serialize the derived metric once attached."""
    summary = {
        'accuracy': 0.45,
        'probe': {'auroc': 0.58},
        'collapse_loss': None,
    }
    add_collapse_loss_to_condition_summary(summary)
    assert math.isclose(summary['collapse_loss'], 0.13, rel_tol=0.0, abs_tol=1e-9)

    payload = json.dumps(summary)
    restored = json.loads(payload)
    assert math.isclose(restored['collapse_loss'], 0.13, rel_tol=0.0, abs_tol=1e-9)
