from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .flowmap import FlowParameters, ScatteredFlowMap, extend_track_forward
from .geometry import DirectedLine, ProjectedROI, load_lines
from .io import file_sha256, write_points_vtp, write_pvd

_worker_model: ScatteredFlowMap | None = None
_worker_table: pd.DataFrame | None = None
_worker_roi: ProjectedROI | None = None
_worker_points_dir: Path | None = None
_worker_summary_dir: Path | None = None
_worker_maximum_duration: int = 0


def _initialize_worker(input_path: str, roi_path: str, flow_values: dict, maximum_duration: int,
                       points_dir: str, summary_dir: str) -> None:
    global _worker_model, _worker_table, _worker_roi, _worker_points_dir, _worker_summary_dir, _worker_maximum_duration
    _worker_table = pd.read_parquet(input_path, columns=["trajid", "time", "x", "y", "z"])
    _worker_model = ScatteredFlowMap(FlowParameters.from_settings(flow_values)).fit(_worker_table)
    _worker_roi = ProjectedROI.from_json(Path(roi_path))
    _worker_points_dir, _worker_summary_dir = Path(points_dir), Path(summary_dir)
    _worker_maximum_duration = maximum_duration


def _process_chunk(chunk_id: int, track_ids: list[int]) -> dict:
    points_path = _worker_points_dir / f"chunk_{chunk_id:05d}.parquet"
    summary_path = _worker_summary_dir / f"chunk_{chunk_id:05d}.parquet"
    if points_path.exists() and summary_path.exists():
        return {"chunk": chunk_id, "tracks": len(track_ids), "points": 0, "skipped": True}
    point_tables, summaries = [], []
    subset = _worker_table[_worker_table["trajid"].isin(track_ids)]
    for track_id, measured in subset.groupby("trajid", sort=False):
        points, summary = extend_track_forward(_worker_model, measured, _worker_roi, _worker_maximum_duration)
        summary["n_virtual_steps"] = int((points["source_type"] == "virtual_lagrangian").sum())
        point_tables.append(points)
        summaries.append(summary)
    combined = pd.concat(point_tables, ignore_index=True) if point_tables else pd.DataFrame()
    combined.to_parquet(points_path, index=False)
    pd.DataFrame(summaries).to_parquet(summary_path, index=False)
    return {"chunk": chunk_id, "tracks": len(summaries), "points": len(combined), "skipped": False}


def run_extension(trajectory_path: Path, run_dir: Path, config: dict, *, track_limit: int | None = None,
                  workers: int | None = None) -> Path:
    settings = config["extension"]
    work = Path(run_dir) / "work" / "lagrangian"
    points_dir, summary_dir = work / "points", work / "summary"
    points_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    ids = np.sort(pd.read_parquet(trajectory_path, columns=["trajid"])["trajid"].unique())
    if track_limit is not None:
        ids = ids[:track_limit]
    chunk_size = int(settings["chunk_size"])
    chunks = [ids[i:i + chunk_size].astype(int).tolist() for i in range(0, len(ids), chunk_size)]
    input_stat = Path(trajectory_path).stat()
    manifest = {
        "trajectory_input": str(Path(trajectory_path).resolve()),
        "trajectory_size_bytes": input_stat.st_size,
        "trajectory_sha256": file_sha256(trajectory_path),
        "roi_sha256": file_sha256(Path(config["geometry"]["roi"])),
        "n_requested_tracks": len(ids),
        "first_track_id": int(ids[0]) if len(ids) else None,
        "last_track_id": int(ids[-1]) if len(ids) else None,
        "flowmap": config["flowmap"], "extension": settings,
    }
    manifest_path = work / "extension_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(
                f"Existing extension partitions in {work} were produced with different input/settings. "
                "Use a new run directory instead of mixing runs."
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    arguments = (str(Path(trajectory_path).resolve()), config["geometry"]["roi"], config["flowmap"],
                 int(settings["maximum_duration_frames"]), str(points_dir.resolve()), str(summary_dir.resolve()))
    worker_count = int(workers if workers is not None else settings["workers"])
    if worker_count == 1:
        _initialize_worker(*arguments)
        for index, chunk in enumerate(chunks):
            result = _process_chunk(index, chunk)
            print(f"extension chunk {index + 1}/{len(chunks)}: {result['tracks']} tracks")
    else:
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_initialize_worker, initargs=arguments) as pool:
            futures = [pool.submit(_process_chunk, index, chunk) for index, chunk in enumerate(chunks)]
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                print(f"extension chunk {completed}/{len(chunks)}: {result['tracks']} tracks")
    summary_files = sorted(summary_dir.glob("chunk_*.parquet"))
    if summary_files:
        pd.concat([pd.read_parquet(path) for path in summary_files], ignore_index=True).to_csv(
            work / "extension_summary.csv", index=False
        )
    return points_dir


def _segment_intersection(x0: np.ndarray, x1: np.ndarray, line: DirectedLine):
    line_vector = line.end - line.start
    path_vector = x1 - x0
    denominator = path_vector[:, 0] * line_vector[1] - path_vector[:, 1] * line_vector[0]
    dx, dy = line.start[0] - x0[:, 0], line.start[1] - x0[:, 1]
    safe = np.where(denominator != 0, denominator, np.nan)
    lam = (dx * line_vector[1] - dy * line_vector[0]) / safe
    mu = (dx * path_vector[:, 1] - dy * path_vector[:, 0]) / safe
    return np.isfinite(lam) & (lam >= 0) & (lam <= 1) & (mu >= 0) & (mu <= 1), lam


def _pair_arrays(points: pd.DataFrame):
    ordered = points.sort_values(["track_id", "time"], kind="mergesort").reset_index(drop=True)
    following = ordered.groupby("track_id")[["x", "y", "z", "time"]].shift(-1)
    mask = following["x"].notna().to_numpy()
    return (ordered, ordered.loc[mask, ["x", "y"]].to_numpy(float),
            following.loc[mask, ["x", "y"]].to_numpy(float), ordered.loc[mask, "track_id"].to_numpy(),
            ordered.loc[mask, "time"].to_numpy(float), following.loc[mask, "time"].to_numpy(float))


def _crossed_track_ids(x0, x1, pair_ids, lines):
    hit = np.zeros(len(x0), dtype=bool)
    for line in lines:
        hit |= _segment_intersection(x0, x1, line)[0]
    return set(np.unique(pair_ids[hit]).tolist())


def _first_directed_exit(x0, x1, pair_ids, t0, lines):
    events = []
    for line in lines:
        hit, lam = _segment_intersection(x0, x1, line)
        vector = line.end - line.start
        side0 = vector[0] * (x0[:, 1] - line.start[1]) - vector[1] * (x0[:, 0] - line.start[0])
        side1 = vector[0] * (x1[:, 1] - line.start[1]) - vector[1] * (x1[:, 0] - line.start[0])
        outward = (side0 > 0) & (side1 <= 0) if line.interior_side == "left" else (side0 < 0) & (side1 >= 0)
        for row in np.flatnonzero(hit & outward):
            events.append((int(pair_ids[row]), float(t0[row] + lam[row]), line.name))
    first: dict[int, str] = {}
    for track_id, _, line_name in sorted(events, key=lambda item: (item[0], item[1])):
        first.setdefault(track_id, line_name)
    return first


def _find_measured_entry(group: pd.DataFrame, lines: list[DirectedLine]) -> str | None:
    xyz = group[["x", "y", "z"]].to_numpy(float)
    if len(xyz) < 2:
        return None
    x0, x1 = xyz[:-1, :2], xyz[1:, :2]
    candidates = []
    for line in lines:
        hit, lam = _segment_intersection(x0, x1, line)
        vector = line.end - line.start
        s0 = vector[0] * (x0[:, 1] - line.start[1]) - vector[1] * (x0[:, 0] - line.start[0])
        s1 = vector[0] * (x1[:, 1] - line.start[1]) - vector[1] * (x1[:, 0] - line.start[0])
        inward = (s0 <= 0) & (s1 > 0) if line.interior_side == "left" else (s0 >= 0) & (s1 < 0)
        indices = np.flatnonzero(hit & inward)
        if len(indices):
            candidates.append((indices[0] + lam[indices[0]], line.name))
    return min(candidates)[1] if candidates else None


def _export_timeseries(table: pd.DataFrame, output_dir: Path, pvd_name: str, series_name: str,
                       label_column: str | None, total_frames: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    series_dir = output_dir / series_name
    series_dir.mkdir(parents=True, exist_ok=True)
    if table.empty:
        pvd = output_dir / pvd_name
        write_pvd(pvd, [])
        return pvd
    table = table.copy()
    table["track_key"] = table["track_id"].astype(str)
    if label_column is not None:
        table["track_key"] = table[label_column].astype(str) + "_" + table["track_key"]
    starts = table.groupby("track_key")["time"].transform("min")
    table["normalized_residence_time"] = ((table["time"] - starts) / total_frames).clip(0, 1)
    table["is_virtual"] = (table["step_index"] >= 0).astype(np.uint8)
    table["frame"] = table["time"].round().astype(int)
    entries = []
    for index, (frame, group) in enumerate(table.groupby("frame", sort=True)):
        name = f"frame_{index:05d}.vtp"
        scalar_columns = ["track_id", "normalized_residence_time", "is_virtual"]
        if label_column is not None:
            scalar_columns.insert(1, label_column)
        write_points_vtp(group, series_dir / name, scalar_columns)
        entries.append((int(frame), f"{series_name}/{name}"))
    pvd = output_dir / pvd_name
    write_pvd(pvd, entries)
    return pvd


def _safe_output_name(name: str, fallback: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "_-" else "_" for character in name)
    return cleaned.strip("_") or fallback


def run_cohorts(trajectory_path: Path, run_dir: Path, config: dict) -> dict[str, Path]:
    work = Path(run_dir) / "work" / "lagrangian"
    points_dir, summary_dir = work / "points", work / "summary"
    points = pd.read_parquet(points_dir)
    ordered, x0, x1, pair_ids, t0, _ = _pair_arrays(points)
    measured = pd.read_parquet(trajectory_path, columns=["trajid", "time", "x", "y", "z"]).rename(columns={"trajid": "track_id"})
    acquisition_frames = int(measured["time"].max() - measured["time"].min() + 1)
    result_dir = Path(run_dir) / "results" / "lagrangian"
    cohort_settings = config["cohorts"]
    enabled = set(cohort_settings.get("enabled_outputs", [
        "chamber_to_exit", "line_to_line", "chamber_stasis", "line_conditions"
    ]))
    outputs: dict[str, Path] = {}
    if "full_paths" in enabled:
        outputs["full_paths"] = _export_timeseries(
            ordered, result_dir, "full_paths_timeseries.pvd", "full_paths_timeseries", None,
            acquisition_frames,
        )
    # This is the explicit no-retrospective-filtering mode: all extended
    # paths are exported, and no chamber/line geometry is even evaluated.
    if enabled == {"full_paths"} or not enabled:
        return outputs

    summary = pd.read_parquet(summary_dir)
    chamber = ProjectedROI.from_json(Path(config["geometry"]["initial_chamber"]))
    crossing_lines = load_lines(Path(config["geometry"]["crossing_lines"]))
    exclusion_lines = load_lines(Path(config["geometry"]["exclusion_lines"]))

    first = ordered.groupby("track_id", sort=False).first()
    chamber_ids = set(first.index[chamber.contains(first[["x", "y", "z"]].to_numpy(float))].tolist())
    first_exit_line = _first_directed_exit(x0, x1, pair_ids, t0, crossing_lines)
    chamber_exit_ids = sorted(chamber_ids.intersection(first_exit_line))
    entry_rows = []
    for track_id, group in measured[measured["track_id"].isin(first_exit_line)].groupby("track_id", sort=False):
        entry = _find_measured_entry(group.sort_values("time"), crossing_lines)
        if entry is not None:
            entry_rows.append({"track_id": int(track_id), "entry_line_id": entry, "exit_line_id": first_exit_line[int(track_id)]})
    line_to_line = pd.DataFrame(entry_rows, columns=["track_id", "entry_line_id", "exit_line_id"])
    line_to_line_ids = line_to_line["track_id"].tolist()

    inside = chamber.contains(ordered[["x", "y", "z"]].to_numpy(float))
    always_inside = pd.Series(inside, index=ordered.index).groupby(ordered["track_id"]).all()
    stasis_ids = sorted(set(always_inside.index[always_inside]).intersection(chamber_ids))

    pd.DataFrame({"track_id": chamber_exit_ids,
                  "exit_line_id": [first_exit_line[i] for i in chamber_exit_ids]}).to_csv(work / "cohort_chamber_to_exit_track_ids.csv", index=False)
    line_to_line.to_csv(work / "cohort_line_to_line_track_ids.csv", index=False)
    pd.DataFrame({"track_id": stasis_ids}).to_csv(work / "cohort_chamber_stasis_track_ids.csv", index=False)
    cohort_frames = []
    seed = int(cohort_settings["random_seed"])
    sample_n = int(cohort_settings["sample_tracks_per_large_cohort"])
    for name, code, ids, sample in [
        ("chamber_to_exit", 0, chamber_exit_ids, True), ("chamber_stasis", 1, stasis_ids, True),
        ("line_to_line", 2, line_to_line_ids, False),
    ]:
        rows = ordered[ordered["track_id"].isin(ids)].copy()
        rows.to_parquet(work / f"cohort_{name}_points.parquet", index=False)
        if name not in enabled:
            continue
        unique = rows["track_id"].unique()
        if sample and len(unique) > sample_n:
            rows = rows[rows["track_id"].isin(np.random.default_rng(seed).choice(unique, sample_n, replace=False))]
        rows["cohort"] = code
        cohort_frames.append(rows)

    all_ids = ordered["track_id"].unique()
    crossed_by_line = {
        line.name: _crossed_track_ids(x0, x1, pair_ids, [line]) for line in crossing_lines
    }
    excluded = _crossed_track_ids(x0, x1, pair_ids, exclusion_lines)
    status = summary.set_index("track_id")["exit_status"]
    first_times = ordered.groupby("track_id")["time"].min()
    first_frame = float(ordered["time"].min())
    diagnostics = pd.DataFrame({"track_id": all_ids})
    for line_name, track_ids in crossed_by_line.items():
        diagnostics[f"cross_{line_name}"] = diagnostics["track_id"].isin(track_ids)
    diagnostics["crosses_excl"] = diagnostics["track_id"].isin(excluded)
    diagnostics["not_exit"] = diagnostics["track_id"].map(status).ne("reconstructed_exit")
    diagnostics["first_frame_present"] = diagnostics["track_id"].map(first_times).eq(first_frame)
    condition_frames = []
    if "line_conditions" in enabled:
        def predicate(token: str) -> pd.Series:
            if token == "no_exit":
                return diagnostics["not_exit"]
            if token == "first_frame":
                return diagnostics["first_frame_present"]
            if token.startswith("cross:"):
                line_name = token.split(":", 1)[1]
                if line_name not in crossed_by_line:
                    raise ValueError(f"Line-cohort rule refers to undefined crossing line {line_name!r}")
                return diagnostics["track_id"].isin(crossed_by_line[line_name])
            raise ValueError(f"Unknown line-cohort predicate: {token!r}")

        for condition, rule in enumerate(cohort_settings.get("line_rules", []), 1):
            rule_id = _safe_output_name(str(rule.get("id", f"rule_{condition}")), f"rule_{condition}")
            mask = pd.Series(True, index=diagnostics.index)
            for token in rule.get("all_of", []):
                mask &= predicate(str(token))
            any_of = [predicate(str(token)) for token in rule.get("any_of", [])]
            if any_of:
                any_mask = any_of[0].copy()
                for item in any_of[1:]:
                    any_mask |= item
                mask &= any_mask
            if bool(rule.get("exclude_crossing", True)):
                mask &= ~diagnostics["crosses_excl"]
            diagnostics[rule_id] = mask
            name = _safe_output_name(str(rule.get("name", rule_id)), rule_id)
            ids = diagnostics.loc[mask, "track_id"]
            rows = ordered[ordered["track_id"].isin(ids)].copy()
            rows.to_parquet(work / f"{name}_points.parquet", index=False)
            rows["condition"] = condition
            condition_frames.append(rows)
    diagnostics.to_csv(work / "line_condition_cohorts.csv", index=False)

    cohort_table = pd.concat(cohort_frames, ignore_index=True) if cohort_frames else pd.DataFrame()
    condition_table = pd.concat(condition_frames, ignore_index=True) if condition_frames else pd.DataFrame()
    if cohort_frames:
        outputs["cohorts"] = _export_timeseries(
            cohort_table, result_dir, "cohorts_timeseries.pvd", "cohorts_timeseries", "cohort",
            acquisition_frames,
        )
    if "line_conditions" in enabled:
        outputs["line_conditions"] = _export_timeseries(
            condition_table, result_dir, "line_condition_timeseries.pvd",
            "line_condition_timeseries", "condition", acquisition_frames,
        )
    return outputs
