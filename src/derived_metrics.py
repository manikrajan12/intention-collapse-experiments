"""
Derived metrics built on top of the core intention-collapse pipeline.

These are empirical summary quantities layered on top of the primary metrics;
they do not replace or rename the core metrics used in the paper.
"""

from typing import Optional, Dict, Any
import math


def normalize_accuracy_for_derived_metrics(accuracy: Optional[float]) -> Optional[float]:
    """
    Normalize accuracy to [0, 1] for derived metric computations.

    The pipeline mostly stores accuracy as a fraction, but some summaries may
    use percentages. Values with absolute magnitude > 1 are treated as
    percentages and divided by 100 before use.
    """
    if accuracy is None:
        return None

    acc = float(accuracy)
    if not math.isfinite(acc):
        return None

    if abs(acc) > 1.0:
        acc /= 100.0

    return acc


def compute_collapse_loss(
    accuracy: Optional[float],
    recoverability_auroc: Optional[float]
) -> Optional[float]:
    """
    Compute collapse loss from final accuracy and recoverability AUROC.

    collapse_loss is an empirical proxy for the gap between recoverable
    pre-collapse signal and final output performance:

        collapse_loss = recoverability_auroc - accuracy

    Higher values mean internal signal exceeds final behavioral performance.
    Negative values mean output performance exceeds the separability implied by
    the probe metric.
    """
    normalized_accuracy = normalize_accuracy_for_derived_metrics(accuracy)
    if normalized_accuracy is None or recoverability_auroc is None:
        return None

    auroc = float(recoverability_auroc)
    if not math.isfinite(auroc):
        return None

    return auroc - normalized_accuracy


def add_collapse_loss_to_condition_summary(condition_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach collapse_loss to a condition summary in place.

    If recoverability is missing for the cell, collapse_loss is set to None.
    """
    probe = condition_summary.get('probe', {}) or {}
    condition_summary['collapse_loss'] = compute_collapse_loss(
        condition_summary.get('accuracy'),
        probe.get('auroc')
    )
    return condition_summary
