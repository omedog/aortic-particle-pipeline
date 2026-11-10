from __future__ import annotations

import numpy as np
import pytest

from aorta_pipeline.synthetic_timescales import (
    GroundTruth,
    brownian_trajectory,
    poiseuille_like_frame,
    turnover_frame,
    uniform_translation_trajectories,
)
from aorta_pipeline.timescales import (
    frame_advection_time,
    frame_diffusion_time,
    frame_residence_ages,
    frame_turnover_time,
    residence_time,
    trajectory_advection_time,
    trajectory_diffusion_coefficient,
    trajectory_diffusion_time,
    transit_time,
)


def test_uniform_translation_residence_and_advection():
    for speed in (0.5, 1.0, 2.5, 7.0):
        for length in (3.0, 10.0, 21.0):
            df, gt = uniform_translation_trajectories(
                speed=speed, length=length, seed=1
            )
            one = df[df["track_id"] == 0].reset_index(drop=True)
            rt = residence_time(one)
            assert rt == pytest.approx(gt.residence_time, rel=1e-9, abs=1e-9)
            tau = trajectory_advection_time(one, length_scale=length)
            assert tau == pytest.approx(gt.advection_time, rel=1e-9, abs=1e-9)


def test_uniform_translation_transit_time():
    df, gt = uniform_translation_trajectories(speed=3.0, length=12.0, seed=2)
    one = df[df["track_id"] == 0].reset_index(drop=True)
    assert transit_time(
        one, entry_time=0.0, exit_time=gt.transit_time
    ) == pytest.approx(gt.transit_time, rel=1e-9)


def test_brownian_diffusion_coefficient_recovery():
    for D in (0.1, 0.5, 1.0, 2.0):
        for seed in (0, 1, 2):
            df, gt = brownian_trajectory(n_steps=5000, diffusivity=D, seed=seed)
            est = trajectory_diffusion_coefficient(df, dims=3)
            assert est == pytest.approx(gt.diffusion_coefficient, rel=0.05)


def test_brownian_diffusion_time():
    df, _gt = brownian_trajectory(n_steps=5000, diffusivity=0.7, seed=3)
    D = trajectory_diffusion_coefficient(df, dims=3)
    tau = trajectory_diffusion_time(df, length_scale=1.0)
    assert tau == pytest.approx(1.0 / D, rel=0.05)


def test_frame_advection_time():
    for speeds in ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [0.5, 4.0]):
        df, gt = poiseuille_like_frame(speeds=np.array(speeds))
        est = frame_advection_time(df, length_scale=1.0)
        assert est == pytest.approx(gt.advection_time, rel=1e-9)


def test_frame_diffusion_time():
    df, gt = poiseuille_like_frame(speeds=np.array([1.0, 3.0, 5.0, 7.0]))
    est = frame_diffusion_time(df, length_scale=1.0)
    assert est == pytest.approx(gt.diffusion_time, rel=1e-9)


def test_frame_turnover_time():
    for volume, L, U in ((100.0, 10.0, 2.0), (50.0, 5.0, 5.0), (1.0, 1.0, 1.0)):
        df, gt = turnover_frame(volume=volume, length_scale=L, normal_speed=U)
        est = frame_turnover_time(df, volume=volume, length_scale=L)
        assert est == pytest.approx(gt.turnover_time, rel=1e-9)


def test_frame_residence_ages_ground_truth():
    frame, _ = poiseuille_like_frame(speeds=np.array([1.0, 2.0, 3.0]))
    frame["time"] = 10.0
    ages = frame_residence_ages(frame, {0: 4.0, 1: 7.0, 2: 9.0})
    np.testing.assert_allclose(ages, [6.0, 3.0, 1.0])


def test_ground_truth_dataclass_defaults():
    gt = GroundTruth()
    assert gt.residence_time is None
    assert gt.turnover_time is None
