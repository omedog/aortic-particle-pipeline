"""Deterministic, physically plausible demo trajectories for pipeline validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

SYNTHETIC_CONFIG = {
    "filtration": {
        "min_length": 20,
        "max_length": 200,
        "minimum_speed_um_per_frame": 0.0,
        "maximum_speed_um_per_frame": 1300.0,
    },
    "eulerian": {
        "grid_shape": [6, 6, 4],
        "minimum_samples": 3,
        "window_frames": 30,
        "step_frames": 10,
    },
    "flowmap": {
        "k_min": 8,
        "k_max": 40,
        "bandwidth_k": 8,
        "minimum_distinct_tracks": 4,
        "minimum_effective_sample_size": 4.0,
        "minimum_confidence": 0.05,
    },
    "extension": {
        "maximum_duration_frames": 40,
        "workers": 1,
        "chunk_size": 16,
    },
    "cohorts": {
        "sample_tracks_per_large_cohort": 500,
        "random_seed": 0,
        "enabled_outputs": ["full_paths", "chamber_to_exit", "line_to_line", "chamber_stasis", "line_conditions"],
    },
    "validation": {
        "minimum_track_length": 40,
        "number_of_tracks": 6,
        "truncation_fractions": [0.25, 0.5, 0.75],
        "maximum_duration_frames": 100,
        "random_seed": 0,
    },
}


def generate_synthetic_dataset(output_dir: Path, *, seed: int = 7) -> dict[str, Path | int]:
    """Create a small staggered ensemble that crosses both internal gates.

    All tracks follow a smooth chamber-to-outlet family while their acquisition
    start frames are staggered. This supplies exact-time spatial neighbours for
    held-out and forward-extension checks without making the demo large.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / "trajectories.zarr"
    config_path = output_dir / "synthetic_settings.json"
    metadata_path = output_dir / "README.txt"

    rng = np.random.default_rng(seed)
    n_tracks, points_per_track = 64, 80
    trajectory_ids, frames, positions = [], [], []
    for track_id in range(n_tracks):
        start_frame = track_id % 32
        age = np.arange(points_per_track, dtype=float)
        progress = age / (points_per_track - 1)
        # A smooth path from the left side of the initial chamber toward
        # the right outlet. It enters through line 1 and exits through line 2.
        base = np.column_stack([
            -0.0105 + 0.0655 * progress,
            0.0400 - 0.0250 * progress + 0.0050 * np.sin(np.pi * progress),
            -0.0300 + 0.0015 * np.sin(2 * np.pi * progress),
        ])
        offset = rng.normal(scale=[1.7e-4, 1.7e-4, 4.0e-4], size=3)
        phase = rng.uniform(0, 2 * np.pi)
        lateral_wobble = np.column_stack([
            np.zeros(points_per_track),
            3e-5 * np.sin(4 * np.pi * progress + phase),
            5e-5 * np.cos(3 * np.pi * progress + phase),
        ])
        trajectory_ids.append(np.full(points_per_track, track_id, dtype=np.int64))
        frames.append(np.arange(start_frame, start_frame + points_per_track, dtype=np.int64))
        positions.append(base + offset + lateral_wobble)

    ids = np.concatenate(trajectory_ids)
    time = np.concatenate(frames)
    position = np.vstack(positions).astype(np.float64)
    order = np.lexsort((time, ids))
    ids, time, position = ids[order], time[order], position[order]

    group = zarr.open_group(str(zarr_path), mode="w")
    row_chunk = min(8192, len(ids))
    group.create_array("trajid", data=ids, chunks=(row_chunk,))
    group.create_array("time", data=time, chunks=(row_chunk,))
    group.create_array("pos", data=position, chunks=(row_chunk, 3))
    group.attrs.update({
        "description": "Deterministic aortic-phantom pipeline validation data",
        "position_units": "m",
        "time_units": "frame",
        "seed": seed,
        "n_tracks": n_tracks,
        "points_per_track": points_per_track,
    })
    config_path.write_text(json.dumps(SYNTHETIC_CONFIG, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(
        "Synthetic aortic-particle validation dataset\n"
        "============================================\n"
        f"Tracks: {n_tracks}\nPoints per track: {points_per_track}\n"
        f"Rows: {len(ids)}\nSeed: {seed}\nPosition units: metres\nTime units: frame\n\n"
        "Use trajectories.zarr as pipeline input and synthetic_settings.json as\n"
        "the configuration override. This dataset is artificial and must not be\n"
        "used for scientific conclusions.\n",
        encoding="utf-8",
    )
    return {
        "zarr": zarr_path,
        "config": config_path,
        "metadata": metadata_path,
        "tracks": n_tracks,
        "rows": len(ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the bundled-style synthetic validation dataset.")
    parser.add_argument("--output", type=Path, default=Path("sample_data"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    created = generate_synthetic_dataset(args.output, seed=args.seed)
    print(f"Synthetic trajectories: {created['zarr']}")
    print(f"Validation settings: {created['config']}")
    print(f"{created['tracks']} tracks, {created['rows']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
