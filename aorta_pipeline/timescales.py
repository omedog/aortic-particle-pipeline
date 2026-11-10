"""Basic transport time scales for Lagrangian particle-tracking data.

The definitions implemented here follow the standard fluid-mechanics and
Lagrangian-PTV literature:

* Residence / exit time and age: Danckwerts (1953), Nauman (2002),
  Deleersnijder et al. (2001).
* Transit time between two sections: Bolin & Rodhe (1973),
  Zimmerman (1976).
* Advection time scale: Tennekes & Lumley (1972), Pope (2000).
* Diffusion / dispersion time scale: Taylor (1922, 1953), Fischer et al.
  (1979), Einstein (1905).
* Turnover time: Levenspiel (1999), Nauman (2002).

Two granularities are provided, matching the two Flowtracks views of a scene:

* **per trajectory** (``Trajectory`` / a track-level table): time scales
  computed along one particle path;
* **per frame** (``ParticleSnapshot`` / a frame-level table): instantaneous,
  bulk time scales computed from all particles present at one time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Literature-cited scalar definitions
# ---------------------------------------------------------------------------


def advection_time_scale(length: float, velocity: float) -> float:
    """Characteristic advection time scale, ``tau_adv = L / U`` (frames or s).

    Tennekes & Lumley (1972), "A First Course in Turbulence"; Pope (2000),
    "Turbulent Flows", Sec. 1.4. ``length`` is the characteristic length of
    the control volume and ``velocity`` is the mean (bulk) velocity.
    """
    length = float(length)
    velocity = float(velocity)
    if not np.isfinite(length) or not np.isfinite(velocity) or velocity == 0.0:
        return float("nan")
    return length / velocity


def diffusion_time_scale(length: float, diffusivity: float) -> float:
    """Characteristic diffusion/dispersion time scale, ``tau_diff = L^2 / D``.

    Taylor (1953), "Dispersion of soluble matter in solvent flowing slowly
    through a tube"; Fischer et al. (1979), "Mixing in Inland and Coastal
    Waters". ``diffusivity`` is the molecular (D) or turbulent (K_t)
    diffusivity in ``length^2 / time``.
    """
    length = float(length)
    diffusivity = float(diffusivity)
    if not np.isfinite(length) or not np.isfinite(diffusivity) or diffusivity <= 0.0:
        return float("nan")
    return length * length / diffusivity


def turnover_time(volume: float, flow_rate: float) -> float:
    """Control-volume turnover (flushing) time, ``tau_turn = V / Q``.

    Levenspiel (1999), "Chemical Reaction Engineering", Chap. 13;
    Nauman (2002), "Residence time theory". ``volume`` is the CV volume and
    ``flow_rate`` is the volumetric flow rate through it.
    """
    volume = float(volume)
    flow_rate = float(flow_rate)
    if not np.isfinite(volume) or not np.isfinite(flow_rate) or flow_rate <= 0.0:
        return float("nan")
    return volume / flow_rate


# ---------------------------------------------------------------------------
# Adapters for Flowtracks objects and plain tables
# ---------------------------------------------------------------------------


def _call_or_attr(obj: Any, name: str):
    """Read ``name`` from an object, calling it if it is a method."""
    if hasattr(obj, name):
        value = getattr(obj, name)
        return value() if callable(value) else value
    return None


def _trajectory_arrays(obj: Any):
    """Return ``(time, pos, velocity_or_None)`` for a trajectory-like object.

    Accepts a pandas DataFrame, a dict, or a Flowtracks ``Trajectory`` (duck
    typed through ``time``/``pos``/``velocity`` attributes or methods).
    """
    if isinstance(obj, pd.DataFrame):
        table = obj.sort_values("time", kind="mergesort").reset_index(drop=True)
        time = table["time"].to_numpy(float)
        pos = table[["x", "y", "z"]].to_numpy(float)
        velocity = None
        for cols in (["vx", "vy", "vz"], ["vx_fd", "vy_fd", "vz_fd"]):
            if all(c in table.columns for c in cols):
                velocity = table[cols].to_numpy(float)
                break
        return time, pos, velocity

    if isinstance(obj, dict):
        time = np.asarray(obj["time"], dtype=float)
        pos = np.asarray(obj["pos"], dtype=float)
        velocity = (
            np.asarray(obj["velocity"], dtype=float) if "velocity" in obj else None
        )
        return time, pos, velocity

    time = _call_or_attr(obj, "time")
    pos = _call_or_attr(obj, "pos")
    velocity = _call_or_attr(obj, "velocity")
    if time is None or pos is None:
        raise TypeError("trajectory object must expose 'time' and 'pos'")
    time = np.asarray(time, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3)
    if velocity is not None:
        velocity = np.asarray(velocity, dtype=float)
        if velocity.ndim == 1:
            velocity = velocity.reshape(-1, 3)
    return time, pos, velocity


def _frame_arrays(obj: Any):
    """Return ``(time_scalar, trajids, pos, velocity_or_None)`` for a frame."""
    if isinstance(obj, pd.DataFrame):
        time_values = obj["time"].to_numpy() if "time" in obj else None
        time_scalar = (
            float(time_values[0])
            if time_values is not None and len(time_values)
            else float("nan")
        )
        trajids = (
            obj["trajid"].to_numpy()
            if "trajid" in obj
            else (
                obj["track_id"].to_numpy() if "track_id" in obj else np.arange(len(obj))
            )
        )
        pos = obj[["x", "y", "z"]].to_numpy(float)
        velocity = None
        for cols in (["vx", "vy", "vz"], ["vx_fd", "vy_fd", "vz_fd"]):
            if all(c in obj.columns for c in cols):
                velocity = obj[cols].to_numpy(float)
                break
        return time_scalar, np.asarray(trajids), pos, velocity

    if isinstance(obj, dict):
        time_scalar = float(obj.get("time", np.nan))
        default_ids = np.arange(len(obj.get("pos", [])))
        trajids = np.asarray(obj.get("trajid", obj.get("trajids", default_ids)))
        pos = np.asarray(obj["pos"], dtype=float)
        velocity = (
            np.asarray(obj["velocity"], dtype=float) if "velocity" in obj else None
        )
        return time_scalar, trajids, pos, velocity

    time = _call_or_attr(obj, "time")
    trajids = _call_or_attr(obj, "trajid")
    pos = _call_or_attr(obj, "pos")
    velocity = _call_or_attr(obj, "velocity")
    if pos is None:
        raise TypeError("frame object must expose 'pos'")
    time_scalar = (
        float(np.asarray(time).reshape(-1)[0])
        if time is not None and np.size(time)
        else float("nan")
    )
    if trajids is None:
        trajids = np.arange(len(np.asarray(pos)))
    pos = np.asarray(pos, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3)
    if velocity is not None:
        velocity = np.asarray(velocity, dtype=float)
        if velocity.ndim == 1:
            velocity = velocity.reshape(-1, 3)
    return time_scalar, np.asarray(trajids), pos, velocity


def _entry_time_of(points) -> float:
    time, _, _ = _trajectory_arrays(points)
    if len(time) == 0:
        return float("nan")
    return float(time[0])


def _exit_time_of(
    points, *, status_column: str = "status", roi_state_column: str = "roi_state"
) -> float:
    """Exit time of an extended track.

    Uses an explicit ``status``/``roi_state`` if present in a DataFrame;
    otherwise falls back to the last available time (measured-only tracks are
    then treated as not having exited).
    """
    if isinstance(points, pd.DataFrame):
        if status_column in points.columns:
            mask = points[status_column].astype(str).str.contains("exit", case=False)
            if mask.any():
                return float(points.loc[mask, "time"].iloc[0])
        if roi_state_column in points.columns:
            mask = points[roi_state_column].astype(str).str.lower().eq("exit")
            if mask.any():
                return float(points.loc[mask, "time"].iloc[0])
        # No exit marker: residence/exit time is unknown (not just track span).
        return float("nan")
    _time, _, _ = _trajectory_arrays(points)
    return float("nan")


def _speeds(points) -> np.ndarray:
    """Per-sample speed (m/frame) for a trajectory."""
    time, pos, velocity = _trajectory_arrays(points)
    if velocity is not None and len(velocity) == len(pos):
        return np.linalg.norm(velocity, axis=1)
    if len(pos) < 2:
        return np.array([], dtype=float)
    dt = np.diff(time)
    dt = np.where(dt == 0, np.nan, dt)
    disp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speeds = disp / dt
    # Centre the speed on samples (interior) for consistency with finite diff.
    out = np.full(len(pos), np.nan)
    out[1:-1] = (speeds[:-1] + speeds[1:]) / 2.0 if len(speeds) > 1 else speeds
    if len(speeds) == 1:
        out[0] = speeds[0]
        out[-1] = speeds[0]
    if len(speeds) > 1:
        out[0] = speeds[0]
        out[-1] = speeds[-1]
    return out


# ---------------------------------------------------------------------------
# Per-trajectory time scales
# ---------------------------------------------------------------------------


def residence_time(
    points,
    *,
    entry_time: float | None = None,
    exit_time: float | None = None,
    status_column: str = "status",
    roi_state_column: str = "roi_state",
) -> float:
    """Residence (exit) time of one trajectory inside the control volume.

    ``t_res = t_exit - t_entry``.  For a track that never exits (censored),
    returns ``NaN``; use :func:`trajectory_duration` for the observed span.

    Literature: Danckwerts (1953), "Continuous flow systems"; Nauman (2002),
    "Residence time theory"; Deleersnijder et al. (2001), "The age as a
    diagnostic of marine life-cycle and transport".
    """
    if entry_time is None:
        entry_time = _entry_time_of(points)
    if exit_time is None:
        exit_time = _exit_time_of(
            points, status_column=status_column, roi_state_column=roi_state_column
        )
    entry_time = float(entry_time)
    exit_time = float(exit_time)
    if not np.isfinite(entry_time) or not np.isfinite(exit_time):
        return float("nan")
    value = exit_time - entry_time
    return value if value >= 0 else float("nan")


def transit_time(
    points,
    *,
    entry_time: float | None = None,
    exit_time: float | None = None,
    status_column: str = "status",
    roi_state_column: str = "roi_state",
) -> float:
    """Transient / transit time between two sections or markers.

    ``t_transit = t_end - t_start``.  Unlike residence time, the two markers
    are arbitrary (e.g. crossing line 1 -> crossing line 2) and must be
    supplied when they are not the first/last sample.

    Literature: Bolin & Rodhe (1973), "A note on the concepts of age
    distribution and transit time in natural reservoirs"; Zimmerman (1976),
    "On the Eulerian and Lagrangian scales of transport".
    """
    if entry_time is None:
        entry_time = _entry_time_of(points)
    if exit_time is None:
        exit_time = _exit_time_of(
            points, status_column=status_column, roi_state_column=roi_state_column
        )
    entry_time = float(entry_time)
    exit_time = float(exit_time)
    if not np.isfinite(entry_time) or not np.isfinite(exit_time):
        return float("nan")
    value = exit_time - entry_time
    return value if value >= 0 else float("nan")


def trajectory_duration(points) -> float:
    """Observed temporal span of a trajectory, ``t_last - t_first`` (frames)."""
    time, _, _ = _trajectory_arrays(points)
    if len(time) < 2:
        return float("nan")
    return float(time[-1] - time[0])


def trajectory_advection_time(points, length_scale: float) -> float:
    """Per-trajectory advection time, ``tau_adv = L / <U>_track``.

    The mean speed is the time average of the particle speed along the
    trajectory.  Literature: Tennekes & Lumley (1972); Pope (2000).
    """
    speeds = _speeds(points)
    speeds = speeds[np.isfinite(speeds)]
    if len(speeds) == 0:
        return float("nan")
    return advection_time_scale(length_scale, float(np.mean(speeds)))


def trajectory_diffusion_coefficient(
    points, *, max_lag: int | None = None, dims: int = 3
) -> float:
    """Lagrangian (eddy) diffusivity from the variance of per-frame increments.

    ``D = Var(dx - <dx>) / (2 * dims * <dt>)``.  Removing the mean increment
    separates turbulent dispersion from the mean advection.  Literature:
    Taylor (1922), "Diffusion by continuous movements"; Einstein (1905).
    """
    time, pos, _velocity = _trajectory_arrays(points)
    if len(pos) < 3:
        return float("nan")
    increments = np.diff(pos, axis=0)
    mean_inc = np.mean(increments, axis=0)
    centered = increments - mean_inc
    msd_inc = float(np.mean(np.sum(centered * centered, axis=1)))
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    mean_dt = float(np.mean(dt)) if len(dt) else 1.0
    if mean_dt <= 0:
        mean_dt = 1.0
    denom = 2.0 * dims * mean_dt
    if denom <= 0:
        return float("nan")
    return msd_inc / denom


def trajectory_diffusion_time(
    points, length_scale: float, *, max_lag: int | None = None, dims: int = 3
) -> float:
    """Per-trajectory diffusion time, ``tau_diff = L^2 / D_track``.

    Literature: Taylor (1953); Fischer et al. (1979).
    """
    diffusivity = trajectory_diffusion_coefficient(points, max_lag=max_lag, dims=dims)
    return diffusion_time_scale(length_scale, diffusivity)


def trajectory_age(points, frame_time: float) -> float:
    """Residence age of the trajectory at ``frame_time``: ``t - t_entry``.

    Literature: Deleersnijder et al. (2001); Sanderson (1991), "Lagrangian
    tracer studies".
    """
    entry = _entry_time_of(points)
    if not np.isfinite(entry):
        return float("nan")
    value = float(frame_time) - entry
    return value if value >= 0 else 0.0


# ---------------------------------------------------------------------------
# Per-frame time scales
# ---------------------------------------------------------------------------


def _lookup(mapping: dict[Any, float], key: Any) -> float | None:
    if key in mapping:
        return mapping[key]
    try:
        key_int = int(key)
    except (TypeError, ValueError):
        return None
    return mapping.get(key_int)


def frame_residence_ages(frame, entry_time_by_track: dict[Any, float]) -> np.ndarray:
    """Per-particle residence age in the control volume at this frame.

    Returns an array aligned with the frame's particles.  Literature:
    Deleersnijder et al. (2001).
    """
    time_scalar, trajids, _, _ = _frame_arrays(frame)
    ages = np.full(len(trajids), np.nan, dtype=float)
    for i, tid in enumerate(trajids):
        entry = _lookup(entry_time_by_track, tid)
        if entry is None:
            continue
        entry = float(entry)
        if np.isfinite(entry) and np.isfinite(time_scalar):
            value = time_scalar - entry
            ages[i] = value if value >= 0 else 0.0
    return ages


def frame_transit_times(frame, reference_time_by_track: dict[Any, float]) -> np.ndarray:
    """Per-particle time elapsed since a reference event (e.g. line crossing).

    Literature: Bolin & Rodhe (1973).
    """
    return frame_residence_ages(frame, reference_time_by_track)


def frame_advection_time(frame, length_scale: float) -> float:
    """Bulk advection time of the particles present in this frame.

    ``tau_adv = L / mean(speed)``.  Literature: Tennekes & Lumley (1972).
    """
    _, _, _, velocity = _frame_arrays(frame)
    if velocity is not None and len(velocity):
        speeds = np.linalg.norm(velocity, axis=1)
    else:
        # Fall back to a zero-length estimate.
        speeds = np.array([], dtype=float)
    speeds = speeds[np.isfinite(speeds)]
    if len(speeds) == 0:
        return float("nan")
    return advection_time_scale(length_scale, float(np.mean(speeds)))


def frame_diffusion_time(frame, length_scale: float) -> float:
    """Bulk diffusion time of the particles present in this frame.

    Turbulent diffusivity is estimated as ``K_t ~ (1/3) * u'_rms * L``
    (Prandtl mixing-length argument), so ``tau_diff = 3 * L / u'_rms``.
    Literature: Prandtl (1925); Taylor (1953); Fischer et al. (1979).
    """
    _, _, _, velocity = _frame_arrays(frame)
    if velocity is None or len(velocity) < 2:
        return float("nan")
    speeds = np.linalg.norm(velocity, axis=1)
    fluctuation = float(np.std(speeds))
    if not np.isfinite(fluctuation) or fluctuation <= 0:
        return float("inf")
    diffusivity = (length_scale * fluctuation) / 3.0
    return diffusion_time_scale(length_scale, diffusivity)


def frame_turnover_time(
    frame,
    volume: float,
    *,
    length_scale: float,
    cross_sectional_area: float | None = None,
) -> float:
    """Instantaneous turnover time of the control volume.

    ``tau_turn = V / Q``.  If a cross-sectional area is given,
    ``Q = A * <U_normal>``; otherwise the flow rate is estimated as
    ``Q = V * <U> / L`` (plug-flow assumption), reducing to the advection
    time scale.  Literature: Levenspiel (1999); Nauman (2002).
    """
    _, _, _, velocity = _frame_arrays(frame)
    if velocity is None or len(velocity) == 0:
        return float("nan")
    mean_speed = float(np.mean(np.linalg.norm(velocity, axis=1)))
    if cross_sectional_area is not None and cross_sectional_area > 0:
        normal = (
            velocity[:, 0]
            if velocity.shape[1] >= 1
            else np.linalg.norm(velocity, axis=1)
        )
        flow_rate = float(cross_sectional_area) * float(np.mean(normal))
    else:
        if length_scale <= 0 or not np.isfinite(mean_speed):
            return float("nan")
        flow_rate = float(volume) * mean_speed / float(length_scale)
    return turnover_time(volume, flow_rate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


__all__ = [
    "advection_time_scale",
    "diffusion_time_scale",
    "frame_advection_time",
    "frame_diffusion_time",
    "frame_residence_ages",
    "frame_transit_times",
    "frame_turnover_time",
    "residence_time",
    "trajectory_advection_time",
    "trajectory_age",
    "trajectory_diffusion_coefficient",
    "trajectory_diffusion_time",
    "trajectory_duration",
    "transit_time",
    "turnover_time",
]
