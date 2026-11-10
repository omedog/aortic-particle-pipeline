from __future__ import annotations

import hashlib
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

POINT_COLUMNS = ["trajid", "time", "x", "y", "z"]
VELOCITY_COLUMNS = ["vx_fd", "vy_fd", "vz_fd", "speed_fd"]


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_trajectories(path: Path) -> pd.DataFrame:
    """Read raw Zarr, Parquet, or CSV trajectories into one canonical table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        source = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        source = pd.read_csv(path)
    else:
        import zarr

        group = zarr.open_group(str(path), mode="r")
        required = {"trajid", "time", "pos"}
        missing = sorted(required.difference(group.keys()))
        if missing:
            raise ValueError(f"Zarr input lacks arrays: {missing}")
        pos = np.asarray(group["pos"][:])
        if pos.ndim != 2 or pos.shape[1] < 3:
            raise ValueError("Zarr 'pos' must have shape (N, 3)")
        source = pd.DataFrame({
            "trajid": np.asarray(group["trajid"][:]),
            "time": np.asarray(group["time"][:]),
            "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        })

    aliases = {"track_id": "trajid", "frame": "time"}
    source = source.rename(columns={k: v for k, v in aliases.items() if k in source and v not in source})
    missing = [c for c in POINT_COLUMNS if c not in source]
    if missing:
        raise ValueError(f"Trajectory input lacks columns: {missing}")
    out = source[POINT_COLUMNS].copy()
    out["trajid"] = out["trajid"].astype(np.int64)
    out["time"] = out["time"].astype(np.int64)
    out[["x", "y", "z"]] = out[["x", "y", "z"]].astype(np.float64)
    return out.sort_values(["trajid", "time"], kind="mergesort").reset_index(drop=True)


def trajectory_summary(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby("trajid", sort=True).agg(
        length=("time", "size"), t_start=("time", "min"), t_end=("time", "max")
    ).reset_index()


def write_trajectory_vtp(table: pd.DataFrame, path: Path) -> None:
    import pyvista as pv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values(["trajid", "time"], kind="mergesort").reset_index(drop=True)
    points = ordered[["x", "y", "z"]].to_numpy(np.float64)
    poly = pv.PolyData()
    poly.points = points
    if len(ordered):
        starts = np.flatnonzero(np.r_[True, ordered["trajid"].to_numpy()[1:] != ordered["trajid"].to_numpy()[:-1]])
        ends = np.r_[starts[1:], len(ordered)]
        lines = np.concatenate([
            np.r_[end - start, np.arange(start, end, dtype=np.int64)]
            for start, end in zip(starts, ends)
        ]).astype(np.int64)
        poly.lines = lines
        poly.point_data["trajid"] = ordered["trajid"].to_numpy(np.int64)
        poly.point_data["time"] = ordered["time"].to_numpy(np.int64)
        if set(VELOCITY_COLUMNS).issubset(ordered.columns):
            poly.point_data["velocity"] = ordered[["vx_fd", "vy_fd", "vz_fd"]].to_numpy(np.float64)
            poly.point_data["speed"] = ordered["speed_fd"].to_numpy(np.float64)
    poly.save(path)


def write_points_vtp(table: pd.DataFrame, path: Path, scalar_columns: list[str]) -> None:
    import pyvista as pv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    poly = pv.PolyData(table[["x", "y", "z"]].to_numpy(np.float64))
    for name in scalar_columns:
        if name in table:
            values = table[name].to_numpy()
            if values.dtype.kind in "OUS":
                values = pd.Categorical(values).codes.astype(np.int32)
            poly.point_data[name] = values
    poly.save(path)


def write_pvd(path: Path, entries: list[tuple[float | int, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for timestep, relative_file in entries:
        lines.append(
            f'    <DataSet timestep="{timestep}" group="" part="0" file="{escape(relative_file)}"/>'
        )
    lines.extend(["  </Collection>", "</VTKFile>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
