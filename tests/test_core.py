from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aorta_pipeline.filtering import finite_difference_velocity
from aorta_pipeline.flowmap import (
    FlowParameters,
    ScatteredFlowMap,
    extend_track_forward,
)
from aorta_pipeline.geometry import ProjectedROI
from aorta_pipeline.settings import load_settings


def test_embedded_defaults_resolve_inside_package():
    settings = load_settings()
    assert all(Path(path).is_file() for path in settings["geometry"].values())


def test_finite_difference_velocity_is_track_local():
    table = pd.DataFrame({
        "trajid": [1, 1, 1, 2, 2], "time": [0, 1, 2, 0, 1],
        "x": [0.0, 1.0, 4.0, 10.0, 12.0], "y": 0.0, "z": 0.0,
    })
    result = finite_difference_velocity(table)
    np.testing.assert_allclose(result.vx_fd, [1.0, 2.0, 3.0, 2.0, 2.0])


def _translation_data(seed=3):
    rng = np.random.default_rng(seed)
    origins = rng.normal(scale=0.04, size=(30, 3)) + [0.75, 0.0, 0.0]
    shift = np.array([0.2, 0.01, -0.01])
    rows = []
    for track_id, origin in enumerate(origins):
        rows.extend([
            {"trajid": track_id, "time": 0, "x": origin[0], "y": origin[1], "z": origin[2]},
            {"trajid": track_id, "time": 1, "x": origin[0] + shift[0], "y": origin[1] + shift[1], "z": origin[2] + shift[2]},
        ])
    return pd.DataFrame(rows), shift


def _test_parameters():
    return FlowParameters(k_min=8, k_max=30, bandwidth_k=8,
                          minimum_distinct_tracks=4, minimum_effective_sample_size=4,
                          minimum_confidence=0.0)


def test_local_affine_recovers_translation():
    table, shift = _translation_data()
    prediction = ScatteredFlowMap(_test_parameters()).fit(table).predict([0.75, 0.0, 0.0], 0)
    assert prediction.support_valid
    np.testing.assert_allclose(prediction.displacement, shift, atol=1e-10)


def test_forward_extension_marks_boundary_exit(tmp_path):
    table, _ = _translation_data()
    model = ScatteredFlowMap(_test_parameters()).fit(table)
    roi_file = tmp_path / "roi.json"
    roi_file.write_text(json.dumps({
        "point_rule": "all_points_inside", "combine_rule": "and",
        "xy_polygon": [[-1, -1], [0.85, -1], [0.85, 1], [-1, 1]],
        "xz_polygon": [[-1, -1], [0.85, -1], [0.85, 1], [-1, 1]],
    }))
    measured = pd.DataFrame([{"trajid": 999, "time": 0, "x": 0.75, "y": 0.0, "z": 0.0}])
    points, summary = extend_track_forward(model, measured, ProjectedROI.from_json(roi_file), 1)
    assert summary["exit_status"] == "reconstructed_exit"
    assert not summary["censored"]
    assert points.iloc[-1].source_type == "virtual_lagrangian"


def test_yz_polygon_is_an_independent_roi_constraint(tmp_path):
    roi_file = tmp_path / "roi_yz.json"
    roi_file.write_text(json.dumps({
        "point_rule": "all_points_inside", "combine_rule": "and",
        "xy_polygon": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
        "xz_polygon": None,
        "yz_polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
    }))
    roi = ProjectedROI.from_json(roi_file)
    assert roi.contains([[0, 0, 0]])[0]
    assert not roi.contains([[0, 1.5, 0]])[0]
