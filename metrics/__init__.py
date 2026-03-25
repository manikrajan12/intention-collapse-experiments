"""Additional analysis metrics for intention-collapse experiments."""

from .rcg import (
    sample_hidden_states,
    compute_recoverability_scores,
    compute_rcg,
    compute_rcg_full,
    plot_rcg_trajectory,
)

__all__ = [
    "sample_hidden_states",
    "compute_recoverability_scores",
    "compute_rcg",
    "compute_rcg_full",
    "plot_rcg_trajectory",
]
