from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

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
    trajectory_diffusion_coefficient,
    trajectory_duration,
    transit_time,
    turnover_time,
)


def test_scalar_edge_cases():
    assert math.isnan(advection_time_scale(1.0, 0.0))
    assert math.isnan(advection_time_scale(1.0, float("nan")))
    assert math.isnan(diffusion_time_scale(1.0, 0.0))
    assert math.isnan(diffusion_time_scale(1.0, -1.0))
    assert math.isnan(turnover_time(1.0, 0.0))
    assert math.isnan(turnover_time(1.0, float("nan")))


def test_short_trajectories():
    empty = pd.DataFrame({"time": [], "x": [], "y": [], "z": []})
    assert math.isnan(trajectory_duration(empty))
    assert math.isnan(trajectory_advection_time(empty, 1.0))
    assert math.isnan(trajectory_diffusion_coefficient(empty))
    single = pd.DataFrame({"time": [0], "x": [0.0], "y": [0.0], "z": [0.0]})
    assert math.isnan(trajectory_duration(single))
    assert math.isnan(trajectory_advection_time(single, 1.0))


def test_residence_time_censored_and_roi_state():
    df = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "x": [0.0, 1.0, 2.0],
            "y": 0.0,
            "z": 0.0,
            "status": ["continued", "continued", "insufficient_lagrangian_support"],
        }
    )
    assert math.isnan(residence_time(df))
    df2 = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "x": [0.0, 1.0, 2.0],
            "y": 0.0,
            "z": 0.0,
            "roi_state": ["inside", "inside", "exit"],
        }
    )
    assert residence_time(df2) == 2.0


def test_residence_time_custom_columns():
    df = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "x": [0.0, 1.0, 2.0],
            "y": 0.0,
            "z": 0.0,
            "state": ["inside", "inside", "exit"],
        }
    )
    assert residence_time(df, status_column="state") == 2.0


def test_transit_time_negative_returns_nan():
    df = pd.DataFrame({"time": [0, 1], "x": [0.0, 1.0], "y": 0.0, "z": 0.0})
    assert math.isnan(transit_time(df, entry_time=5.0, exit_time=3.0))


def test_trajectory_age_clipping():
    df = pd.DataFrame({"time": [0, 1], "x": [0.0, 1.0], "y": 0.0, "z": 0.0})
    assert trajectory_age(df, -1.0) == 0.0
    assert trajectory_age(df, 0.5) == 0.5


def test_dict_adapter():
    traj = {
        "time": np.array([0.0, 1.0, 2.0]),
        "pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "velocity": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    assert trajectory_duration(traj) == 2.0
    assert trajectory_advection_time(traj, 2.0) == 2.0
    frame = {
        "time": 5.0,
        "trajid": np.array([1, 2]),
        "pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "velocity": np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    }
    assert frame_advection_time(frame, 2.0) == 2.0 / 2.0


def test_frame_residence_ages_missing_entry():
    frame = pd.DataFrame(
        {
            "time": [5.0, 5.0],
            "trajid": [1, 2],
            "x": [0.0, 1.0],
            "y": 0.0,
            "z": 0.0,
        }
    )
    ages = frame_residence_ages(frame, {1: 2.0})
    assert ages[0] == 3.0
    assert math.isnan(ages[1])


def test_frame_diffusion_zero_fluctuation():
    frame = pd.DataFrame(
        {
            "time": [1.0, 1.0],
            "trajid": [1, 2],
            "x": [0.0, 1.0],
            "y": 0.0,
            "z": 0.0,
            "vx": [2.0, 2.0],
            "vy": 0.0,
            "vz": 0.0,
        }
    )
    assert frame_diffusion_time(frame, 1.0) == float("inf")


def test_frame_turnover_with_area():
    frame = pd.DataFrame(
        {
            "time": [1.0, 1.0],
            "trajid": [1, 2],
            "x": [0.0, 1.0],
            "y": 0.0,
            "z": 0.0,
            "vx": [2.0, 2.0],
            "vy": 0.0,
            "vz": 0.0,
        }
    )
    # V=10, L=2, A=5, U=2 => Q=10 => tau=1
    assert frame_turnover_time(
        frame, volume=10.0, length_scale=2.0, cross_sectional_area=5.0
    ) == pytest.approx(1.0)


def test_diffusion_coefficient_with_dt():
    # dt=2 frames, D=0.5 => per-coord variance = 2*D*dt = 2.0
    rng = np.random.default_rng(0)
    n = 20000
    inc = rng.normal(scale=np.sqrt(2.0), size=(n, 3))
    pos = np.concatenate([np.zeros((1, 3)), np.cumsum(inc, axis=0)], axis=0)
    time = np.arange(n + 1) * 2.0
    df = pd.DataFrame({"time": time, "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]})
    est = trajectory_diffusion_coefficient(df, dims=3)
    assert est == pytest.approx(0.5, rel=0.05)
