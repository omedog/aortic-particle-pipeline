from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import ProjectedROI
from .io import read_trajectories, trajectory_summary, write_trajectory_vtp


def finite_difference_velocity(table):
    table = table.sort_values(["trajid", "time"], kind="mergesort").reset_index(drop=True).copy()
    ids = table["trajid"].to_numpy()
    positions = table[["x", "y", "z"]].to_numpy(float)
    velocity = np.empty_like(positions)
    if len(table):
        first = np.r_[True, ids[1:] != ids[:-1]]
        last = np.r_[ids[:-1] != ids[1:], True]
        singleton = first & last
        interior = ~(first | last)
        interior_indices = np.flatnonzero(interior)
        velocity[interior_indices] = (
            positions[interior_indices + 1] - positions[interior_indices - 1]
        ) / 2.0
        first_indices = np.flatnonzero(first & ~singleton)
        last_indices = np.flatnonzero(last & ~singleton)
        velocity[first_indices] = positions[first_indices + 1] - positions[first_indices]
        velocity[last_indices] = positions[last_indices] - positions[last_indices - 1]
        velocity[singleton] = np.nan
    table[["vx_fd", "vy_fd", "vz_fd"]] = velocity
    table["speed_fd"] = np.linalg.norm(velocity, axis=1)
    return table


def run_filtration(raw_input: Path, run_dir: Path, config: dict) -> dict[str, Path]:
    settings = config["filtration"]
    minimum_length = int(settings["min_length"])
    maximum_length = int(settings["max_length"])
    if minimum_length < 2 or minimum_length > maximum_length:
        raise ValueError("length thresholds must satisfy 2 <= minimum <= maximum")
    lower_speed = float(settings.get("minimum_speed_um_per_frame", 0.0)) / 1e6
    upper_speed = float(settings.get(
        "maximum_speed_um_per_frame", settings.get("speed_cutoff_um_per_frame", 1300.0)
    )) / 1e6
    if lower_speed < 0 or lower_speed > upper_speed:
        raise ValueError("speed thresholds must satisfy 0 <= minimum <= maximum")
    work_dir = Path(run_dir) / "work" / "filtered"
    result_dir = Path(run_dir) / "results" / "filtered_trajects"  # retain published folder spelling
    work_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    raw = read_trajectories(Path(raw_input))
    if raw.duplicated(["trajid", "time"]).any():
        raise ValueError("Raw input contains duplicate (trajid, time) observations")
    finite = np.isfinite(raw[["x", "y", "z"]]).all(axis=1)
    if not finite.all():
        raise ValueError(f"Raw input contains {int((~finite).sum())} non-finite positions")

    lengths = raw.groupby("trajid").size()
    keep = lengths.index[(lengths >= minimum_length) & (lengths <= maximum_length)]
    length_filtered = finite_difference_velocity(raw[raw["trajid"].isin(keep)].copy())
    if length_filtered.empty:
        raise ValueError("No trajectories remain after length filtration")
    length_filtered.to_parquet(work_dir / "length_filtered.parquet", index=False)
    trajectory_summary(length_filtered).to_csv(work_dir / "length_filtered_summary.csv", index=False)
    write_trajectory_vtp(length_filtered, result_dir / "filtered_trajectories.vtp")

    speed_filtered = length_filtered[
        length_filtered["speed_fd"].between(lower_speed, upper_speed, inclusive="both")
    ].copy()
    post_lengths = speed_filtered.groupby("trajid").size()
    keep = post_lengths.index[post_lengths >= minimum_length]
    speed_filtered = speed_filtered[speed_filtered["trajid"].isin(keep)].copy()

    roi = ProjectedROI.from_json(Path(config["geometry"]["roi"]))
    speed_filtered["_inside"] = roi.contains(speed_filtered[["x", "y", "z"]].to_numpy(float))
    keep_track = speed_filtered.groupby("trajid")["_inside"].transform("all")
    final = speed_filtered.loc[keep_track].drop(columns="_inside").reset_index(drop=True)
    if final.empty:
        raise ValueError("No trajectories remain after speed and ROI filtration")
    final.to_parquet(work_dir / "trajectories.parquet", index=False)
    trajectory_summary(final).to_csv(work_dir / "trajectory_summary.csv", index=False)
    write_trajectory_vtp(final, result_dir / "roi_filtered_trajectories.vtp")

    report = {
        "raw_points": len(raw), "raw_tracks": int(raw["trajid"].nunique()),
        "length_filtered_points": len(length_filtered),
        "length_filtered_tracks": int(length_filtered["trajid"].nunique()),
        "final_points": len(final), "final_tracks": int(final["trajid"].nunique()),
        "length_range_points": [minimum_length, maximum_length],
        "speed_range_um_per_frame": [lower_speed * 1e6, upper_speed * 1e6],
    }
    (work_dir / "filtration_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"trajectories": work_dir / "trajectories.parquet", "results": result_dir}
