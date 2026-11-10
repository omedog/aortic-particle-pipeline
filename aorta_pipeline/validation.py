from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .flowmap import FlowParameters, ScatteredFlowMap, extend_track_forward
from .geometry import ProjectedROI


def run_validation(trajectory_path: Path, run_dir: Path, config: dict) -> Path:
    """Artificially truncate held-out measured tracks and quantify chained error."""
    settings = config["validation"]
    raw = pd.read_parquet(trajectory_path, columns=["trajid", "time", "x", "y", "z"])
    canonical = raw.rename(columns={"trajid": "track_id"}).sort_values(["track_id", "time"])
    lengths = canonical.groupby("track_id").size()
    candidates = lengths[lengths >= int(settings["minimum_track_length"])].index.to_numpy()
    rng = np.random.default_rng(int(settings["random_seed"]))
    selected = np.sort(rng.choice(candidates, min(int(settings["number_of_tracks"]), len(candidates)), replace=False))
    support = raw[~raw["trajid"].isin(selected)]
    model = ScatteredFlowMap(FlowParameters.from_settings(config["flowmap"])).fit(support)
    roi = ProjectedROI.from_json(Path(config["geometry"]["roi"]))
    records, cases = [], []
    maximum_duration = int(settings["maximum_duration_frames"])

    for track_id in selected:
        full = canonical[canonical["track_id"] == track_id].reset_index(drop=True)
        positions = full[["x", "y", "z"]].to_numpy(float)
        step_displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        typical_step = float(step_displacements.mean()) if len(step_displacements) else np.nan
        for fraction in settings["truncation_fractions"]:
            cutoff = max(1, min(len(full) - 2, round(float(fraction) * len(full))))
            known, hidden = full.iloc[:cutoff + 1].copy(), full.iloc[cutoff + 1:].copy()
            reconstructed, summary = extend_track_forward(model, known, roi, maximum_duration)
            virtual = reconstructed[reconstructed["source_type"] == "virtual_lagrangian"]
            truth = hidden.set_index("time")[["x", "y", "z"]]
            cutoff_time, hidden_end = float(known["time"].iloc[-1]), float(hidden["time"].iloc[-1])
            matched = []
            for _, row in virtual.iterrows():
                frame = round(row["time"])
                if frame not in truth.index:
                    continue
                actual = truth.loc[frame].to_numpy(float)
                if actual.ndim > 1:
                    actual = actual[0]
                error = float(np.linalg.norm(actual - row[["x", "y", "z"]].to_numpy(float)))
                normalized = error / typical_step if typical_step > 0 else np.nan
                elapsed = float(row["time"] - cutoff_time)
                matched.append((elapsed, error, normalized))
                records.append({"track_id": int(track_id), "fraction": float(fraction),
                                "elapsed_since_cutoff": elapsed, "position_error": error,
                                "position_error_norm": normalized})
            duration = hidden_end - cutoff_time
            coverage = max((row[0] for row in matched), default=0.0) / duration if duration > 0 else 0.0
            cases.append({
                "track_id": int(track_id), "fraction": float(fraction), "known_length": len(known),
                "hidden_length": len(hidden), "hidden_duration_frames": duration,
                "status": summary["exit_status"], "n_matched_frames": len(matched),
                "coverage_fraction": coverage, "track_mean_frame_displacement": typical_step,
                "final_position_error": matched[-1][1] if matched else np.nan,
                "final_position_error_norm": matched[-1][2] if matched else np.nan,
                "mean_position_error": np.mean([row[1] for row in matched]) if matched else np.nan,
                "mean_position_error_norm": np.mean([row[2] for row in matched]) if matched else np.nan,
            })

    result_dir = Path(run_dir) / "results" / "validation"
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    errors = pd.DataFrame(records, columns=["track_id", "fraction", "elapsed_since_cutoff",
                                           "position_error", "position_error_norm"])
    pd.DataFrame(cases).to_csv(result_dir / "case_summaries.csv", index=False)
    errors.to_csv(result_dir / "pointwise_position_errors.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 6), facecolor="white")
    if len(errors):
        upper = float(errors["elapsed_since_cutoff"].quantile(0.98))
        if upper > 0:
            bins = np.linspace(0, upper, 30)
            labels = pd.cut(errors["elapsed_since_cutoff"], np.unique(bins))
            stats = errors.groupby(labels, observed=True)["position_error_norm"].agg(
                median="median", p25=lambda values: values.quantile(0.25), p75=lambda values: values.quantile(0.75)
            )
            centers = np.array([interval.mid for interval in stats.index])
            ax.fill_between(centers, stats.p25.to_numpy(float), stats.p75.to_numpy(float),
                            color="#0072B2", alpha=0.2, label="25th–75th percentile")
            ax.plot(centers, stats["median"].to_numpy(float), color="#0072B2", linewidth=2.2, label="median")
            ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No reconstructed frames matched held-out measurements",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("elapsed time since truncation (frames)")
    ax.set_ylabel("position error / track mean per-frame displacement")
    ax.set_title("Reconstruction accuracy vs. time since cutoff")
    ax.grid(True, color="#d8d8d8", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output = figure_dir / "error_vs_elapsed_time.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return result_dir
