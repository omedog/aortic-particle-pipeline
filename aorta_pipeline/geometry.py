from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath


@dataclass(frozen=True)
class ProjectedROI:
    xy_polygon: np.ndarray | None
    xz_polygon: np.ndarray | None
    yz_polygon: np.ndarray | None = None

    @classmethod
    def from_json(cls, path: Path) -> ProjectedROI:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("point_rule", "all_points_inside") != "all_points_inside":
            raise ValueError("Only the all_points_inside geometry rule is supported")
        if data.get("combine_rule", "and") != "and":
            raise ValueError("Only the AND projection rule is supported")
        xy = np.asarray(data["xy_polygon"], dtype=float) if data.get("xy_polygon") else None
        xz = np.asarray(data["xz_polygon"], dtype=float) if data.get("xz_polygon") else None
        yz = np.asarray(data["yz_polygon"], dtype=float) if data.get("yz_polygon") else None
        if xy is None and xz is None and yz is None:
            raise ValueError(f"No polygons in {path}")
        return cls(xy, xz, yz)

    def contains(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        inside = np.ones(len(points), dtype=bool)
        if self.xy_polygon is not None:
            inside &= PolygonPath(self.xy_polygon).contains_points(points[:, [0, 1]])
        if self.xz_polygon is not None:
            inside &= PolygonPath(self.xz_polygon).contains_points(points[:, [0, 2]])
        if self.yz_polygon is not None:
            inside &= PolygonPath(self.yz_polygon).contains_points(points[:, [1, 2]])
        return inside

    def segment_exit(self, start: np.ndarray, end: np.ndarray, iterations: int = 30) -> np.ndarray | None:
        if not self.contains(start)[0]:
            return np.asarray(start, dtype=float)
        if self.contains(end)[0]:
            return None
        lo, hi = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        for _ in range(iterations):
            mid = (lo + hi) / 2.0
            if self.contains(mid)[0]:
                lo = mid
            else:
                hi = mid
        # Return a point just beyond the boolean flip so a repeated
        # containment test consistently classifies it as outside.
        direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        return hi + 1e-6 * direction


@dataclass(frozen=True)
class DirectedLine:
    name: str
    start: np.ndarray
    end: np.ndarray
    interior_side: str

    def side(self, xy: np.ndarray) -> float:
        delta = self.end - self.start
        point = np.asarray(xy) - self.start
        return float(delta[0] * point[1] - delta[1] * point[0])

    def crosses_inward(self, a: np.ndarray, b: np.ndarray) -> bool:
        before, after = self.side(a[:2]), self.side(b[:2])
        if self.interior_side == "left":
            return before <= 0 < after
        return before >= 0 > after


def load_lines(path: Path) -> list[DirectedLine]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("lines", data) if isinstance(data, dict) else data
    return [DirectedLine(
        str(row.get("id", row.get("name", f"line_{i + 1}"))), np.asarray(row["start"], float),
        np.asarray(row["end"], float), str(row.get("interior_side", row.get("side", "left"))).lower(),
    ) for i, row in enumerate(rows)]
