"""Mesh-free, time-resolved local-affine flow map and forward extension.

This is deliberately independent of the Eulerian grid: every virtual step is
estimated directly from measured particle pairs at the matching frame.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import ProjectedROI


@dataclass(frozen=True)
class FlowParameters:
    lag_frames: int = 1
    k_min: int = 20
    k_max: int = 150
    bandwidth_k: int = 20
    bandwidth_scale_factor: float = 1.0
    truncation_multiplier: float = 3.0
    regularization_lambda: float = 1e-6
    fallback_expand_factor: float = 1.5
    fallback_max_expansions: int = 3
    fallback_regularization_multiplier: float = 10.0
    minimum_distinct_tracks: int = 8
    minimum_effective_sample_size: float = 10.0
    maximum_condition_number: float = 1e8
    extrapolation_factor: float = 3.0
    minimum_confidence: float = 0.2
    maximum_step_displacement_factor: float = 10.0
    crossing_bisection_iterations: int = 30

    @classmethod
    def from_settings(cls, values: dict) -> FlowParameters:
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


@dataclass
class NeighborSet:
    indices: np.ndarray
    weights: np.ndarray
    distances: np.ndarray
    bandwidth: float
    radius: float


class NeighborIndex:
    def __init__(self, points: np.ndarray, parameters: FlowParameters):
        self.points = np.asarray(points, dtype=float)
        self.parameters = parameters
        self.n = len(self.points)
        self.tree = cKDTree(self.points)
        if self.n < 2:
            self.reference_bandwidth = 1e-12
        else:
            k = min(parameters.bandwidth_k + 1, self.n)
            distances, _ = self.tree.query(self.points, k=k)
            kth = np.atleast_2d(distances)[:, -1]
            positive = kth[kth > 0]
            self.reference_bandwidth = float(np.median(positive)) if len(positive) else 1e-12

    def query(self, point: np.ndarray) -> NeighborSet:
        p = self.parameters
        if not self.n:
            return NeighborSet(np.array([], int), np.array([]), np.array([]), 1e-12, 0.0)
        k = min(p.bandwidth_k, self.n)
        distances, _ = self.tree.query(point, k=k)
        ordered = np.sort(np.atleast_1d(distances))
        positive = ordered[ordered > 0]
        kth = ordered[-1] if ordered[-1] > 0 else (positive[0] if len(positive) else 1e-12)
        bandwidth = max(float(p.bandwidth_scale_factor * kth), 1e-300)
        radius = p.truncation_multiplier * bandwidth
        indices = np.asarray(self.tree.query_ball_point(point, radius), dtype=int)
        for _ in range(50):
            if len(indices) >= p.k_min or len(indices) >= self.n:
                break
            radius *= 1.5
            indices = np.asarray(self.tree.query_ball_point(point, radius), dtype=int)
        return self._result(point, indices, bandwidth, radius)

    def expand(self, previous: NeighborSet, point: np.ndarray) -> NeighborSet:
        radius = previous.radius * self.parameters.fallback_expand_factor
        indices = np.asarray(self.tree.query_ball_point(point, radius), dtype=int) if self.n else np.array([], int)
        return self._result(point, indices, previous.bandwidth, radius)

    def _result(self, point, indices, bandwidth, radius):
        if len(indices):
            distances = np.linalg.norm(self.points[indices] - point, axis=1)
            if len(indices) > self.parameters.k_max:
                order = np.argsort(distances)[:self.parameters.k_max]
                indices, distances = indices[order], distances[order]
            weights = np.exp(-(distances ** 2) / (2 * bandwidth ** 2))
        else:
            distances, weights = np.array([]), np.array([])
        return NeighborSet(indices, weights, distances, float(bandwidth), float(radius))


@dataclass
class Prediction:
    endpoint: np.ndarray
    displacement: np.ndarray
    gradient: np.ndarray
    n_obs: int
    n_tracks: int
    effective_sample_size: float
    weighted_rmse: float
    bandwidth: float
    condition_number: float
    rank: int
    dist_nearest: float
    support_valid: bool
    fallback_mode: str
    confidence_score: float


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.sum(weights ** 2))
    return float(np.sum(weights) ** 2 / denominator) if denominator > 0 else 0.0


def _affine(origin, displacement, weights, query, regularization):
    dim = origin.shape[1]
    offsets = origin - query
    design = np.c_[np.ones(len(origin)), offsets]
    root_weights = np.sqrt(np.maximum(weights, 0))
    weighted_design = design * root_weights[:, None]
    ridge = np.zeros((dim, dim + 1))
    ridge[:, 1:] = np.eye(dim) * np.sqrt(regularization)
    matrix = np.r_[weighted_design, ridge]
    coefficients = np.empty((dim + 1, dim))
    rank = 0
    for component in range(dim):
        rhs = np.r_[displacement[:, component] * root_weights, np.zeros(dim)]
        coefficients[:, component], _, component_rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        rank = max(rank, int(component_rank))
    d0 = coefficients[0]
    gradient = coefficients[1:].T
    residuals = np.linalg.norm(displacement - (d0 + offsets @ gradient.T), axis=1)
    condition = float(np.linalg.cond(weighted_design)) if len(origin) else np.inf
    rmse = float(np.sqrt(np.sum(weights * residuals ** 2) / np.sum(weights))) if np.sum(weights) > 0 else np.nan
    return d0, gradient, condition, rank, rmse


def _invalid(query: np.ndarray, bandwidth=1e-12, distance=np.inf) -> Prediction:
    dim = len(query)
    return Prediction(np.full(dim, np.nan), np.full(dim, np.nan), np.full((dim, dim), np.nan),
                      0, 0, 0.0, np.nan, bandwidth, np.nan, 0, distance, False, "invalid", 0.0)


def _fit_prediction(index, origin, displacement, track_ids, query, parameters):
    neighborhood = index.query(query)
    support_set = neighborhood
    for _ in range(parameters.fallback_max_expansions):
        if len(support_set.distances) and support_set.distances.min() <= parameters.extrapolation_factor * index.reference_bandwidth:
            break
        support_set = index.expand(support_set, query)
    nearest = float(support_set.distances.min()) if len(support_set.distances) else np.inf
    if nearest > parameters.extrapolation_factor * index.reference_bandwidth:
        return _invalid(query, support_set.bandwidth, nearest)

    def solve(current, regularization, mode, ignore_condition=False):
        selected = current.indices
        if not len(selected):
            return None
        d0, gradient, condition, rank, rmse = _affine(
            origin[selected], displacement[selected], current.weights, query, regularization
        )
        tracks = int(np.unique(track_ids[selected]).size)
        ess = _effective_sample_size(current.weights)
        adequate = rank == len(query) + 1 and tracks >= parameters.minimum_distinct_tracks and ess >= parameters.minimum_effective_sample_size
        if not ignore_condition:
            adequate &= condition <= parameters.maximum_condition_number
        return (adequate, d0, gradient, condition, rank, rmse, tracks, ess, current, mode)

    candidates = [solve(neighborhood, parameters.regularization_lambda, "affine")]
    expanded = neighborhood
    for _ in range(parameters.fallback_max_expansions):
        expanded = index.expand(expanded, query)
        candidates.append(solve(expanded, parameters.regularization_lambda, "affine_expanded"))
    candidates.append(solve(
        expanded, parameters.regularization_lambda * parameters.fallback_regularization_multiplier,
        "affine_regularized", True,
    ))
    accepted = next((candidate for candidate in candidates if candidate and candidate[0]), None)
    if accepted is None:
        selected = expanded.indices
        tracks = int(np.unique(track_ids[selected]).size) if len(selected) else 0
        ess = _effective_sample_size(expanded.weights)
        if tracks < max(2, parameters.minimum_distinct_tracks // 2) or ess < parameters.minimum_effective_sample_size / 2:
            return _invalid(query, expanded.bandwidth, nearest)
        weights = expanded.weights
        d0 = np.average(displacement[selected], axis=0, weights=weights)
        residuals = np.linalg.norm(displacement[selected] - d0, axis=1)
        rmse = float(np.sqrt(np.average(residuals ** 2, weights=weights)))
        gradient, condition, rank, mode = np.zeros((len(query), len(query))), np.nan, 1, "constant"
        current = expanded
    else:
        _, d0, gradient, condition, rank, rmse, tracks, ess, current, mode = accepted

    support_denominator = parameters.extrapolation_factor * current.bandwidth
    support_term = float(np.clip(1 - nearest / support_denominator, 0, 1)) if support_denominator > 0 else 0.0
    sample_term = float(np.clip(ess / 40.0, 0, 1))
    fit_term = 0.5 if np.isnan(condition) else float(1 / (1 + condition / parameters.maximum_condition_number))
    confidence = 0.4 * support_term + 0.3 * sample_term + 0.3 * fit_term
    return Prediction(query + d0, d0, gradient, len(current.indices), tracks, ess, rmse,
                      current.bandwidth, condition, rank, nearest, True, mode, confidence)


class ScatteredFlowMap:
    """Exact-lag flow-map observations with an LRU of per-frame KD trees."""

    def __init__(self, parameters: FlowParameters, cache_size: int = 128):
        self.parameters = parameters
        self.cache_size = cache_size
        self._frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._indices: OrderedDict[int, NeighborIndex] = OrderedDict()

    def fit(self, trajectories: pd.DataFrame) -> ScatteredFlowMap:
        required = {"track_id", "time", "x", "y", "z"}
        table = trajectories.rename(columns={"trajid": "track_id"}).copy()
        if missing := sorted(required.difference(table.columns)):
            raise ValueError(f"Flow-map input lacks columns: {missing}")
        table = table[["track_id", "time", "x", "y", "z"]].sort_values(["track_id", "time"])
        if table.duplicated(["track_id", "time"]).any():
            raise ValueError("Duplicate flow-map observations")
        if not np.isfinite(table[["x", "y", "z"]].to_numpy()).all():
            raise ValueError("Non-finite flow-map observations")
        lag = self.parameters.lag_frames
        left = table.copy()
        left["end_time"] = left["time"] + lag
        right = table.rename(columns={"time": "end_time", "x": "x1", "y": "y1", "z": "z1"})
        pairs = left.merge(right, on=["track_id", "end_time"], how="inner", sort=False)
        for frame, group in pairs.groupby("time", sort=False):
            origin = group[["x", "y", "z"]].to_numpy(float)
            endpoint = group[["x1", "y1", "z1"]].to_numpy(float)
            self._frames[int(frame)] = (origin, endpoint - origin, group["track_id"].to_numpy())
        self._indices.clear()
        return self

    def predict(self, point: np.ndarray, start_time: int) -> Prediction:
        query = np.asarray(point, dtype=float).reshape(-1)
        arrays = self._frames.get(int(start_time))
        if arrays is None:
            return _invalid(query)
        if start_time not in self._indices:
            self._indices[start_time] = NeighborIndex(arrays[0], self.parameters)
            if len(self._indices) > self.cache_size:
                self._indices.popitem(last=False)
        else:
            self._indices.move_to_end(start_time)
        return _fit_prediction(self._indices[start_time], *arrays, query, self.parameters)


def extend_track_forward(model: ScatteredFlowMap, measured: pd.DataFrame, roi: ProjectedROI,
                         maximum_duration: int) -> tuple[pd.DataFrame, dict]:
    measured = measured.rename(columns={"trajid": "track_id"}).sort_values("time").reset_index(drop=True)
    track_id = int(measured["track_id"].iloc[0])
    positions = measured[["x", "y", "z"]].to_numpy(float)
    times = measured["time"].to_numpy(float)
    measured_rows = []
    for _, row in measured.iterrows():
        measured_rows.append({"track_id": track_id, "step_index": -1, "time": float(row.time),
                              "x": row.x, "y": row.y, "z": row.z, "source_type": "measured",
                              "direction": "measured", "n_tracks": np.nan, "effective_sample_size": np.nan,
                              "dist_nearest": np.nan, "confidence_score": np.nan, "fallback_mode": "",
                              "support_valid": True, "roi_state": "inside", "status": "measured", "censored": False})

    for i in range(len(positions) - 1):
        if roi.contains(positions[i])[0] and not roi.contains(positions[i + 1])[0]:
            return pd.DataFrame(measured_rows[:i + 1]), {
                "track_id": track_id, "exit_status": "measured_exit", "censored": False,
                "measured_duration": float(times[i] - times[0]), "virtual_duration": 0.0,
                "fraction_reconstructed": 0.0, "exit_time": float(times[i]),
            }

    point, time = positions[-1].copy(), float(times[-1])
    status, exit_time = "maximum_duration_reached", np.nan
    virtual_rows = []
    step_dt = model.parameters.lag_frames
    for step_index in range(maximum_duration // step_dt):
        prediction = model.predict(point, round(time))
        if not prediction.support_valid or prediction.confidence_score < model.parameters.minimum_confidence:
            status = "insufficient_lagrangian_support"
            virtual_rows.append({
                "track_id": track_id, "step_index": step_index, "time": float(time + step_dt),
                "x": point[0], "y": point[1], "z": point[2], "source_type": "virtual_lagrangian",
                "direction": "forward", "n_tracks": prediction.n_tracks,
                "effective_sample_size": prediction.effective_sample_size,
                "dist_nearest": prediction.dist_nearest, "confidence_score": prediction.confidence_score,
                "fallback_mode": prediction.fallback_mode, "support_valid": prediction.support_valid,
                "roi_state": "inside", "status": status, "censored": True,
            })
            break
        endpoint = prediction.endpoint
        if not np.isfinite(endpoint).all() or np.linalg.norm(endpoint - point) > model.parameters.maximum_step_displacement_factor * prediction.bandwidth:
            status = "numerical_failure"
            virtual_rows.append({
                "track_id": track_id, "step_index": step_index, "time": float(time + step_dt),
                "x": endpoint[0], "y": endpoint[1], "z": endpoint[2], "source_type": "virtual_lagrangian",
                "direction": "forward", "n_tracks": prediction.n_tracks,
                "effective_sample_size": prediction.effective_sample_size,
                "dist_nearest": prediction.dist_nearest, "confidence_score": prediction.confidence_score,
                "fallback_mode": prediction.fallback_mode, "support_valid": prediction.support_valid,
                "roi_state": "inside", "status": status, "censored": True,
            })
            break
        boundary = roi.segment_exit(point, endpoint, model.parameters.crossing_bisection_iterations)
        is_exit = boundary is not None
        next_point = boundary if is_exit else endpoint
        fraction = (np.linalg.norm(next_point - point) / np.linalg.norm(endpoint - point)) if np.linalg.norm(endpoint - point) else 1.0
        next_time = time + step_dt * fraction
        step_status = "reconstructed_exit" if is_exit else "continued"
        virtual_rows.append({
            "track_id": track_id, "step_index": step_index, "time": float(next_time),
            "x": next_point[0], "y": next_point[1], "z": next_point[2],
            "source_type": "virtual_lagrangian", "direction": "forward",
            "n_tracks": prediction.n_tracks, "effective_sample_size": prediction.effective_sample_size,
            "dist_nearest": prediction.dist_nearest, "confidence_score": prediction.confidence_score,
            "fallback_mode": prediction.fallback_mode, "support_valid": prediction.support_valid,
            "roi_state": "exit" if is_exit else "inside", "status": step_status, "censored": False,
        })
        point, time = next_point, next_time
        if is_exit:
            status, exit_time = "reconstructed_exit", next_time
            break

    censored = status not in {"measured_exit", "reconstructed_exit"}
    measured_duration = float(times[-1] - times[0])
    virtual_duration = max(0.0, time - float(times[-1]))
    total = measured_duration + virtual_duration
    fraction_reconstructed = virtual_duration / total if total > 0 else 0.0
    points = pd.DataFrame(measured_rows + virtual_rows)
    points["fraction_reconstructed"] = fraction_reconstructed
    return points, {
        "track_id": track_id, "exit_status": status, "censored": censored,
        "measured_duration": measured_duration, "virtual_duration": virtual_duration,
        "fraction_reconstructed": fraction_reconstructed, "exit_time": exit_time,
    }
