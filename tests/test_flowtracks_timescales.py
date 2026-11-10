from __future__ import annotations

import numpy as np
import pytest

flowtracks = pytest.importorskip("flowtracks")
from flowtracks.io import save_zarr_trajectories
from flowtracks.trajectory import Trajectory
from flowtracks.zarr_scene import ZarrScene

from aorta_pipeline.flowtracks_timescales import (
    flowtracks_available,
    scene_frame_time_scales,
    scene_trajectory_time_scales,
    snapshot_time_scales,
    trajectory_time_scales,
)


def _make_trajectories():
    tr1 = Trajectory(
        pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        velocity=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        time=np.array([0, 1, 2]),
        trajid=1,
    )
    tr2 = Trajectory(
        pos=np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]]),
        velocity=np.array([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]),
        time=np.array([0, 1, 2]),
        trajid=2,
    )
    return [tr1, tr2]


@pytest.fixture
def scene(tmp_path):
    trajects = _make_trajectories()
    zarr_path = tmp_path / "ref.zarr"
    save_zarr_trajectories(trajects, zarr_path)
    return ZarrScene(zarr_path), trajects


def test_flowtracks_available():
    assert flowtracks_available() is True


def test_trajectory_time_scales(scene):
    _, trajects = scene
    result = trajectory_time_scales(trajects[0], length_scale=2.0)
    assert result["duration"] == 2.0
    assert result["advection_time"] == pytest.approx(2.0 / 1.0)
    assert result["diffusion_coefficient"] == pytest.approx(0.0, abs=1e-12)


def test_snapshot_time_scales(scene):
    sc, _ = scene
    snap = next(sc.iter_frames())
    result = snapshot_time_scales(snap, length_scale=2.0, volume=8.0)
    assert "advection_time" in result
    assert "turnover_time" in result
    assert np.isfinite(result["advection_time"])


def test_snapshot_residence_ages(scene):
    sc, _ = scene
    snap = next(sc.iter_frames())
    trajids = snap.trajid().tolist()
    entry_times = {tid: 0.0 for tid in trajids}
    result = snapshot_time_scales(snap, length_scale=2.0, entry_times=entry_times)
    assert "residence_ages" in result
    assert len(result["residence_ages"]) == len(trajids)


def test_scene_trajectory_time_scales(scene):
    sc, _ = scene
    rows = list(scene_trajectory_time_scales(sc, length_scale=2.0))
    assert len(rows) == 2
    assert {r["trajid"] for r in rows} == {1, 2}


def test_scene_frame_time_scales(scene):
    sc, _ = scene
    rows = list(scene_frame_time_scales(sc, length_scale=2.0, volume=8.0))
    assert len(rows) >= 1
    assert all("time" in r and "turnover_time" in r for r in rows)


def test_scene_frame_time_scales_no_volume(scene):
    sc, _ = scene
    rows = list(scene_frame_time_scales(sc, length_scale=2.0))
    assert len(rows) >= 1
    assert all("turnover_time" not in r for r in rows)
