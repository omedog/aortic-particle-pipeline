from __future__ import annotations

import numpy as np
import pandas as pd

from aorta_pipeline.timescales import (
    advection_time_scale,
    diffusion_time_scale,
    frame_advection_time,
    frame_diffusion_time,
    frame_residence_ages,
    frame_turnover_time,
    residence_time,
    trajectory_advection_time,
    trajectory_age,
    trajectory_duration,
    transit_time,
    turnover_time,
)


def test_scalar_definitions():
    assert advection_time_scale(10.0, 2.0) == 5.0
    assert diffusion_time_scale(10.0, 2.0) == 50.0
    assert turnover_time(100.0, 5.0) == 20.0


def test_trajectory_duration_and_age():
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 3],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": 0.0,
            "z": 0.0,
        }
    )
    assert trajectory_duration(df) == 3.0
    assert trajectory_age(df, 5.0) == 5.0


def test_trajectory_advection_time():
    df = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "x": [0.0, 1.0, 2.0],
            "y": 0.0,
            "z": 0.0,
        }
    )
    # mean speed = 1 m/frame -> tau_adv = 10 / 1 = 10 frames
    assert trajectory_advection_time(df, 10.0) == 10.0


def test_residence_time_with_status():
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 3],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": 0.0,
            "z": 0.0,
            "status": ["continued", "continued", "reconstructed_exit", "continued"],
        }
    )
    assert residence_time(df) == 2.0
    assert transit_time(df, entry_time=0.0, exit_time=2.0) == 2.0


def test_frame_functions():
    frame = pd.DataFrame(
        {
            "time": [5, 5],
            "trajid": [1, 2],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "vx": [1.0, 3.0],
            "vy": [0.0, 0.0],
            "vz": [0.0, 0.0],
        }
    )
    ages = frame_residence_ages(frame, {1: 2.0, 2: 4.0})
    np.testing.assert_allclose(ages, [3.0, 1.0])
    assert frame_advection_time(frame, 10.0) == 10.0 / 2.0
    assert np.isfinite(frame_diffusion_time(frame, 10.0))
    assert np.isfinite(frame_turnover_time(frame, 100.0, length_scale=10.0))
