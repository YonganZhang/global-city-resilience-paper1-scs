from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.ablation import normalised_improvement
from experiments.external_validation import pairwise_order_concordance
from experiments.peer_control import dispersion_reductions, pair_peer_arms, paired_cell_metrics
from experiments.repeatability import compare_runs
from experiments.sensitivity import LOCAL_GRID


def test_local_sensitivity_grid_has_27_arms_and_canonical() -> None:
    assert len(LOCAL_GRID) == 27
    assert (0.8, 0.3, 0.3) in LOCAL_GRID


def test_peer_control_uses_paired_models_and_mean_sd_reduction() -> None:
    key_rows = [
        {"city": "A", "indicator": "x", "model": "m1", "score": 0.0},
        {"city": "A", "indicator": "x", "model": "m2", "score": 10.0},
    ]
    r1 = pd.DataFrame(key_rows)
    real = r1.assign(score=[2.5, 7.5])
    scrambled = r1.assign(score=[0.1, 9.9])
    metrics = paired_cell_metrics(pair_peer_arms(r1, real, scrambled))
    reduction = dispersion_reductions(metrics)
    assert np.isclose(reduction["real_reduction_vs_r1"], 0.5)
    assert np.isclose(reduction["scrambled_reduction_vs_r1"], 0.02)


def test_repeatability_and_pairwise_order_statistics() -> None:
    baseline = pd.DataFrame(
        {"city": ["A", "B", "C", "D"], "gciri": [4.0, 3.0, 2.0, 1.0], "cluster": ["P1", "P2", "P3", "P4"]}
    )
    repeat = baseline.assign(gciri=[4.1, 2.9, 2.1, 0.9])
    result = compare_runs(baseline, repeat)
    assert np.isclose(result["pearson_r"], 0.9970544855015817)
    assert np.isclose(result["mean_absolute_difference"], 0.1)
    assert result["mode_changes"] == 0
    assert pairwise_order_concordance(baseline["gciri"], repeat["gciri"]) == 1.0


def test_ablation_normalisation() -> None:
    assert normalised_improvement(2.0, full_metric=1.0, one_agent_metric=5.0) == 75.0
