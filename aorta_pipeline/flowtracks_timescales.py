"""Optional Flowtracks integration for the generic time-scale API.

This module is intentionally dependency-light: it imports ``flowtracks`` lazily
and raises a clear error only when a Flowtracks-specific helper is actually
used without the package installed.  The core estimators in
:mod:`aorta_pipeline.timescales` remain usable on plain tables and duck-typed
objects without Flowtracks.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from .timescales import (
    frame_advection_time,
    frame_diffusion_time,
    frame_residence_ages,
    frame_turnover_time,
    residence_time,
    trajectory_advection_time,
    trajectory_age,
    trajectory_diffusion_coefficient,
    trajectory_diffusion_time,
    trajectory_duration,
    transit_time,
)


def flowtracks_available() -> bool:
    """Return ``True`` when the optional ``flowtracks`` package is importable."""
    try:
        import flowtracks  # noqa: F401
    except ImportError:
        return False
    return True


def _require_flowtracks():
    try:
        import flowtracks
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "flowtracks is required for this helper. Install the optional "
            "dependency or use the generic tables API in aorta_pipeline.timescales."
        ) from exc
    return flowtracks


def trajectory_time_scales(
    traj: Any,
    length_scale: float,
    *,
    entry_time: float | None = None,
    exit_time: float | None = None,
) -> dict[str, float]:
    """Compute all basic time scales for one Flowtracks ``Trajectory``.

    Returns a dictionary with ``duration``, ``residence_time``, ``transit_time``,
    ``advection_time``, ``diffusion_coefficient``, ``diffusion_time`` and
    ``age_at_last``.  Literature references are in
    :mod:`aorta_pipeline.timescales`.
    """
    _require_flowtracks()
    rt = residence_time(traj, entry_time=entry_time, exit_time=exit_time)
    tt = transit_time(traj, entry_time=entry_time, exit_time=exit_time)
    return {
        "duration": trajectory_duration(traj),
        "residence_time": rt,
        "transit_time": tt,
        "advection_time": trajectory_advection_time(traj, length_scale),
        "diffusion_coefficient": trajectory_diffusion_coefficient(traj),
        "diffusion_time": trajectory_diffusion_time(traj, length_scale),
        "age_at_last": trajectory_age(traj, float(np.asarray(traj.time())[-1])),
    }


def snapshot_time_scales(
    snapshot: Any,
    length_scale: float,
    *,
    volume: float | None = None,
    cross_sectional_area: float | None = None,
    entry_times: dict[Any, float] | None = None,
) -> dict[str, Any]:
    """Compute bulk time scales for one Flowtracks ``ParticleSnapshot``.

    Returns advection, diffusion and (if ``volume`` is given) turnover times,
    plus per-particle residence ages when ``entry_times`` is supplied.
    """
    _require_flowtracks()
    result: dict[str, Any] = {
        "advection_time": frame_advection_time(snapshot, length_scale),
        "diffusion_time": frame_diffusion_time(snapshot, length_scale),
    }
    if volume is not None:
        result["turnover_time"] = frame_turnover_time(
            snapshot,
            volume,
            length_scale=length_scale,
            cross_sectional_area=cross_sectional_area,
        )
    if entry_times is not None:
        result["residence_ages"] = frame_residence_ages(snapshot, entry_times)
    return result


def scene_trajectory_time_scales(
    scene: Any, length_scale: float
) -> Iterator[dict[str, Any]]:
    """Yield per-trajectory time scales for every trajectory in a scene."""
    _require_flowtracks()
    for traj in scene.iter_trajectories():
        yield {
            "trajid": int(traj.trajid()),
            **trajectory_time_scales(traj, length_scale),
        }


def scene_frame_time_scales(
    scene: Any,
    length_scale: float,
    *,
    volume: float | None = None,
    cross_sectional_area: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield per-frame bulk time scales for every frame in a scene."""
    _require_flowtracks()
    for snap in scene.iter_frames():
        entry_times = None
        yield {
            "time": float(snap.time()),
            **snapshot_time_scales(
                snap,
                length_scale,
                volume=volume,
                cross_sectional_area=cross_sectional_area,
                entry_times=entry_times,
            ),
        }


__all__ = [
    "flowtracks_available",
    "scene_frame_time_scales",
    "scene_trajectory_time_scales",
    "snapshot_time_scales",
    "trajectory_time_scales",
]
