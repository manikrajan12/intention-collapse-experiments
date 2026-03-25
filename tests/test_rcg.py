import math
import sys
from pathlib import Path

import numpy as np

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from metrics.rcg import (
    sample_hidden_states,
    compute_recoverability_scores,
    compute_rcg,
    compute_rcg_full,
)
from src.shared_utils import ExperimentResult, IntentionMetrics, attach_rcg_to_results


class DummyProbe:
    def predict_proba(self, X):
        X = np.asarray(X)
        positive = np.clip(X[:, 0], 0.0, 1.0)
        negative = 1.0 - positive
        return np.stack([negative, positive], axis=1)


def _make_result(problem_idx: int) -> ExperimentResult:
    return ExperimentResult(
        problem_idx=problem_idx,
        original_dataset_idx=problem_idx,
        condition="baseline",
        benchmark="gsm8k",
        question="q",
        ground_truth="1",
        model_output="#### 1",
        extracted_answer="1",
        is_correct=True,
        metrics=IntentionMetrics(entropy=0.0),
    )


def test_sample_hidden_states_includes_last_token():
    hidden_states = np.arange(30, dtype=float).reshape(10, 3)
    sampled = sample_hidden_states(hidden_states, step_size=4)
    assert sampled.shape == (4, 3)
    assert np.array_equal(sampled[-1], hidden_states[-1])


def test_compute_recoverability_scores_uses_positive_probability():
    sampled_states = np.array([[0.1, 0.0], [0.7, 0.0], [1.0, 0.0]])
    scores = compute_recoverability_scores(sampled_states, DummyProbe())
    assert scores == [0.1, 0.7, 1.0]


def test_compute_rcg_returns_mean_delta():
    rcg = compute_rcg([0.2, 0.5, 0.6])
    assert math.isclose(rcg, 0.2, rel_tol=0.0, abs_tol=1e-9)
    assert compute_rcg([0.4]) == 0.0


def test_compute_rcg_full_batches_sampling_and_scoring():
    hidden_states = np.array([
        [0.1, 0.0],
        [0.2, 0.0],
        [0.5, 0.0],
        [0.8, 0.0],
        [0.9, 0.0],
    ])
    payload = compute_rcg_full(hidden_states, DummyProbe(), step_size=2)
    assert payload["trajectory"] == [0.1, 0.5, 0.9]
    assert payload["num_steps"] == 3
    assert math.isclose(payload["rcg"], 0.4, rel_tol=0.0, abs_tol=1e-9)


def test_attach_rcg_to_results_updates_examples_and_returns_summary():
    results = [_make_result(0), _make_result(1)]
    activation_arrays = [
        np.array([[0.1, 0.0], [0.6, 0.0], [0.8, 0.0]]),
        np.array([[0.8, 0.0], [0.4, 0.0], [0.2, 0.0]]),
    ]
    summary = attach_rcg_to_results(
        results,
        activation_arrays,
        activation_idxs=[0, 1],
        probe_model=DummyProbe(),
        step_size=1,
        trajectory_max_points=2,
        debug_examples=0,
    )

    assert "rcg_mean" in summary
    assert math.isclose(results[0].rcg, 0.35, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(results[1].rcg, -0.3, rel_tol=0.0, abs_tol=1e-9)
    assert results[0].rcg_trajectory == [0.1, 0.6]
    assert results[0].rcg_num_steps == 3
    assert math.isclose(summary["rcg_positive_rate"], 0.5, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(summary["rcg_negative_rate"], 0.5, rel_tol=0.0, abs_tol=1e-9)
