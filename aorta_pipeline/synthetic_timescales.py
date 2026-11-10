"""Synthetic, ground-truth data for validating transport time-scale estimators.

These generators are deliberately flow-agnostic: they produce plain tables with
the same columns used throughout :mod:`aorta_pipeline.timescales` and expose
the exact expected time scales, so tests can check numerical recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroundTruth:
    """Expected values associated with a synthetic dataset."""

    residence_time: float | None = None
    transit_time: float | None = None
    advection_time: float | None = None
    diffusion_coefficient: float | None = None
    diffusion_time: float | None = None
    turnover_time: float | None = None


def uniform_translation_trajectories(
    *,
    n_tracks: int = 8,
    speed: float = 2.0,
    length: float = 10.0,
    dt: float = 1.0,
    lateral_jitter: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, GroundTruth]:
    """Tracks moving at constant speed along +x from 0 to ``length``.

    Expected: ``residence_time = transit_time = length / speed`` and
    ``advection_time = length / speed``. Diffusivity is ~0.
    """
    rng = np.random.default_rng(seed)
    rows = []
    exit_time = length / speed
    # Intermediate sample times strictly before the exit; final sample at exit.
    intermediate = np.arange(0.0, exit_time, dt)
    times = np.concatenate([intermediate, [exit_time]])
    times = np.unique(np.round(times, 9))
    for track_id in range(n_tracks):
        for time in times:
            x = speed * time if time < exit_time else length
            x = min(x, length)
            y = rng.normal(scale=lateral_jitter)
            z = rng.normal(scale=lateral_jitter)
            is_exit = abs(time - exit_time) < 1e-9
            status = "reconstructed_exit" if is_exit and time > 0 else "continued"
            rows.append(
                {
                    "track_id": track_id,
                    "trajid": track_id,
                    "time": float(time),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "status": status,
                    "roi_state": "exit" if status == "reconstructed_exit" else "inside",
                }
            )
    frame = pd.DataFrame(rows)
    expected = GroundTruth(
        residence_time=length / speed,
        transit_time=length / speed,
        advection_time=length / speed,
        diffusion_coefficient=0.0,
    )
    return frame, expected


def brownian_trajectory(
    *,
    n_steps: int = 2000,
    diffusivity: float = 0.5,
    dims: int = 3,
    dt: float = 1.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, GroundTruth]:
    """Isotropic Brownian motion with a known diffusion coefficient.

    Each coordinate increment has variance ``2 * D * dt``, so the total
    squared displacement per step has expectation ``2 * dims * D * dt``.
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2.0 * diffusivity * dt)
    increments = rng.normal(scale=sigma, size=(n_steps, dims))
    positions = np.concatenate(
        [np.zeros((1, dims)), np.cumsum(increments, axis=0)], axis=0
    )
    times = np.arange(n_steps + 1, dtype=float) * dt
    frame = pd.DataFrame(
        {
            "trajid": 0,
            "time": times,
            "x": positions[:, 0],
            "y": positions[:, 1] if dims > 1 else 0.0,
            "z": positions[:, 2] if dims > 2 else 0.0,
        }
    )
    expected = GroundTruth(diffusion_coefficient=diffusivity)
    return frame, expected


def poiseuille_like_frame(
    *,
    speeds: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, GroundTruth]:
    """A single frame with particles whose speeds are known.

    Expected advection time uses the mean speed; diffusion time uses the
    speed fluctuation.
    """
    if speeds is None:
        speeds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    speeds = np.asarray(speeds, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(speeds)
    frame = pd.DataFrame(
        {
            "time": np.full(n, 7.0),
            "trajid": np.arange(n),
            "x": rng.uniform(size=n),
            "y": rng.uniform(size=n),
            "z": rng.uniform(size=n),
            "vx": speeds,
            "vy": np.zeros(n),
            "vz": np.zeros(n),
        }
    )
    mean_speed = float(np.mean(speeds))
    fluctuation = float(np.std(speeds))
    length_scale = 1.0
    expected = GroundTruth(
        advection_time=length_scale / mean_speed,
        diffusion_time=(3.0 * length_scale / fluctuation)
        if fluctuation > 0
        else float("inf"),
    )
    return frame, expected


def turnover_frame(
    *,
    volume: float = 120.0,
    length_scale: float = 10.0,
    normal_speed: float = 4.0,
    n: int = 20,
    seed: int = 0,
) -> tuple[pd.DataFrame, GroundTruth]:
    """A frame whose turnover time is known from V and Q=A*U."""
    rng = np.random.default_rng(seed)
    area = volume / length_scale
    flow_rate = area * normal_speed
    frame = pd.DataFrame(
        {
            "time": np.full(n, 3.0),
            "trajid": np.arange(n),
            "x": rng.uniform(size=n),
            "y": rng.uniform(size=n),
            "z": rng.uniform(size=n),
            "vx": np.full(n, normal_speed),
            "vy": np.zeros(n),
            "vz": np.zeros(n),
        }
    )
    expected = GroundTruth(turnover_time=volume / flow_rate)
    return frame, expected


__all__ = [
    "GroundTruth",
    "brownian_trajectory",
    "poiseuille_like_frame",
    "turnover_frame",
    "uniform_translation_trajectories",
]
