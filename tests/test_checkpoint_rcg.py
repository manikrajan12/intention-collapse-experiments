import sys
from pathlib import Path
import shutil

import numpy as np

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.checkpoint_utils import ExperimentRunner


def test_experiment_runner_saves_rcg_trajectories():
    tmpdir = root_dir / "results" / "test_checkpoint_rcg_tmp"
    if tmpdir.exists():
        shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        runner = ExperimentRunner(
            run_id="testrun",
            output_dir=str(tmpdir),
            condition="baseline",
            sync_interval=1,
        )
        runner.start(total_items=1)
        runner.set_current_idx(0)

        runner.save_item(
            {"problem_idx": 0, "is_correct": True},
            {
                "prompt_activations": np.zeros((2, 3), dtype=np.float32),
                "trajectory_activations": np.ones((4, 6), dtype=np.float32),
            },
        )
        summary = runner.finalize()

        acts_path = Path(summary["activations_path"])
        assert acts_path.exists()

        data = np.load(str(acts_path), allow_pickle=True)
        assert "activations" in data
        assert "rcg_activations" in data
        assert "idxs" in data
        assert data["activations"].shape == (1, 2, 3)
        assert len(data["rcg_activations"]) == 1
        assert data["rcg_activations"][0].shape == (4, 6)
        assert data["idxs"].tolist() == [0]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
