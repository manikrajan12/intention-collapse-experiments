"""
Reasoning Consistency Gain (RCG).

RCG measures whether hidden states become progressively more aligned with the
correct final answer during generation. It uses an existing probe model to
score sampled hidden states along the trajectory, then averages the stepwise
score gains.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np


def _to_numpy(array_like: Any) -> np.ndarray:
    """Convert numpy / torch inputs to a CPU numpy array."""
    if isinstance(array_like, np.ndarray):
        return array_like

    detach = getattr(array_like, "detach", None)
    if callable(detach):
        array_like = detach()

    cpu = getattr(array_like, "cpu", None)
    if callable(cpu):
        array_like = cpu()

    return np.asarray(array_like)


def sample_hidden_states(hidden_states: Any, step_size: int) -> np.ndarray:
    """
    Sample hidden states every `step_size` tokens while always keeping the end.

    Args:
        hidden_states: Array shaped [num_tokens, hidden_dim] or compatible.
        step_size: Sampling interval in tokens. Must be positive.

    Returns:
        Sampled hidden states with the final token always included.
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    hidden_states = _to_numpy(hidden_states)
    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have at least 2 dimensions")
    if hidden_states.shape[0] == 0:
        return hidden_states

    sample_idxs = list(range(0, hidden_states.shape[0], step_size))
    final_idx = hidden_states.shape[0] - 1
    if not sample_idxs or sample_idxs[-1] != final_idx:
        sample_idxs.append(final_idx)

    return hidden_states[sample_idxs]


def compute_recoverability_scores(sampled_states: Any, probe_model: Any) -> List[float]:
    """
    Score sampled states with an existing probe model.

    Uses batched prediction for efficiency. If `predict_proba` is available,
    returns the positive-class probability, which corresponds to correctness
    for the existing probe setup. Otherwise falls back to decision scores or
    raw predictions.
    """
    sampled_states = _to_numpy(sampled_states)
    if sampled_states.ndim < 2:
        raise ValueError("sampled_states must have at least 2 dimensions")
    if sampled_states.shape[0] == 0:
        return []

    flat_states = sampled_states.reshape(sampled_states.shape[0], -1)

    if hasattr(probe_model, "predict_proba"):
        probs = probe_model.predict_proba(flat_states)
        probs = np.asarray(probs)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1].astype(float).tolist()
        return probs.reshape(-1).astype(float).tolist()

    if hasattr(probe_model, "decision_function"):
        scores = np.asarray(probe_model.decision_function(flat_states))
        return scores.reshape(-1).astype(float).tolist()

    scores = np.asarray(probe_model.predict(flat_states))
    return scores.reshape(-1).astype(float).tolist()


def compute_rcg(S_t: List[float]) -> float:
    """
    Compute mean stepwise gain across a recoverability trajectory.

    Returns 0.0 for degenerate trajectories with fewer than two points.
    """
    if len(S_t) < 2:
        return 0.0

    scores = np.asarray(S_t, dtype=float)
    deltas = scores[1:] - scores[:-1]
    return float(np.mean(deltas))


def compute_rcg_full(hidden_states: Any, probe_model: Any, step_size: int = 20) -> dict:
    """Compute sampled trajectory scores and RCG from hidden states."""
    sampled_states = sample_hidden_states(hidden_states, step_size=step_size)
    trajectory = compute_recoverability_scores(sampled_states, probe_model)
    rcg = compute_rcg(trajectory)
    return {
        "rcg": float(rcg),
        "trajectory": trajectory,
        "num_steps": len(trajectory),
    }


def plot_rcg_trajectory(S_t: List[float]):
    """Simple debug plot for an RCG score trajectory."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(len(S_t)), S_t, marker="o", linewidth=1.5)
    ax.set_xlabel("Sample Step")
    ax.set_ylabel("Recoverability Score")
    ax.set_title("RCG Trajectory")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
