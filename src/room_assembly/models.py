"""Coordinate-explicit data contracts. No file reads or session persistence."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections.abc import Mapping
from typing import Any
import numpy as np
from shapely.geometry import Polygon

from .core.candidates import Edge
from .errors import AssemblyError


def finite_number(value, field_name):
    if isinstance(value, (bool, str, bytes)):
        raise AssemblyError("INVALID_INPUT", f"{field_name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssemblyError("INVALID_INPUT", f"{field_name} must be a finite number") from exc
    if not math.isfinite(result):
        raise AssemblyError("INVALID_INPUT", f"{field_name} must be a finite number")
    return result


@dataclass(frozen=True)
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta_deg: float = 0.0

    def __post_init__(self):
        for name in ("x", "y", "theta_deg"):
            object.__setattr__(self, name, finite_number(getattr(self, name), name))

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {"x", "y", "theta_deg"}:
            raise AssemblyError("INVALID_INPUT", "pose must contain exactly x, y, theta_deg")
        return cls(**value)

    def radians(self):
        return self.x, self.y, math.radians(self.theta_deg)

    def to_dict(self):
        return {"x": self.x, "y": self.y, "theta_deg": self.theta_deg}


@dataclass
class AssemblyState:
    placed: dict[str, Pose2D] = field(default_factory=dict)
    excluded_pair_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.placed, Mapping):
            raise AssemblyError("INVALID_INPUT", "placed must map room IDs to poses")
        self.placed = {room: Pose2D.parse(pose) for room, pose in self.placed.items()}
        if not all(isinstance(room, str) and room for room in self.placed):
            raise AssemblyError("INVALID_INPUT", "room IDs must be nonempty strings")
        if not isinstance(self.excluded_pair_ids, (list, tuple)) or not all(isinstance(i, str) for i in self.excluded_pair_ids):
            raise AssemblyError("INVALID_INPUT", "excluded_pair_ids must be a sequence of strings")
        self.excluded_pair_ids = tuple(self.excluded_pair_ids)

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            # Copy so a caller's mutable state never becomes engine-owned state.
            return cls(dict(value.placed), tuple(value.excluded_pair_ids))
        if not isinstance(value, Mapping) or "placed" not in value or set(value) - {"placed", "excluded_pair_ids"}:
            raise AssemblyError("INVALID_INPUT", "state needs placed and optional excluded_pair_ids")
        return cls(**value)

    def to_dict(self):
        return {"placed": {k: v.to_dict() for k, v in self.placed.items()},
                "excluded_pair_ids": list(self.excluded_pair_ids)}


@dataclass
class Scene:
    scene_id: str
    polygons: dict[str, Any]
    edges: list[Edge] = field(default_factory=list)
    root_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise AssemblyError("INVALID_INPUT", "scene_id must be a nonempty string")
        if not isinstance(self.polygons, Mapping) or not self.polygons:
            raise AssemblyError("INVALID_INPUT", "at least one room polygon is required")
        converted = {}
        for room, raw in self.polygons.items():
            if not isinstance(room, str) or not room:
                raise AssemblyError("INVALID_INPUT", "room IDs must be nonempty strings")
            try:
                points = np.array(raw, dtype=float, copy=True)
            except (TypeError, ValueError) as exc:
                raise AssemblyError("INVALID_INPUT", "polygon must be numeric Nx2", room_id=room) from exc
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
                raise AssemblyError("INVALID_INPUT", "polygon must be finite Nx2 with at least 3 vertices", room_id=room)
            if np.linalg.norm(points[0] - points[-1]) <= 1e-9:
                raise AssemblyError("INVALID_INPUT", "do not repeat the first vertex; corner indices must remain stable", room_id=room)
            shape = Polygon(points)
            if not shape.is_valid or shape.area <= 1e-9 or np.any(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1) <= 1e-9):
                raise AssemblyError("INVALID_INPUT", "polygon must be simple, nonzero and have no zero-length walls", room_id=room)
            points.setflags(write=False)
            converted[room] = points
        self.polygons = converted
        if not isinstance(self.edges, (list, tuple)) or not all(isinstance(edge, Edge) for edge in self.edges):
            raise AssemblyError("INVALID_INPUT", "edges must contain Edge objects from a scene adapter")
        seen = set()
        for edge in self.edges:
            if edge.room_a not in converted or edge.room_b not in converted or edge.room_a == edge.room_b:
                raise AssemblyError("INVALID_INPUT", "pair references invalid rooms", pair_index=edge.index)
            if not isinstance(edge.index, int) or isinstance(edge.index, bool) or edge.index < 0 or edge.index in seen:
                raise AssemblyError("INVALID_INPUT", "pair indices must be unique nonnegative integers")
            seen.add(edge.index)
            if not isinstance(edge.measurement, (list, tuple)) or len(edge.measurement) != 3:
                raise AssemblyError("INVALID_INPUT", "edge measurement must have three components")
            for value in (*edge.measurement, edge.confidence, edge.automatic_confidence):
                finite_number(value, "edge pose/score")
            if not 0 <= edge.confidence <= 1 or not 0 <= edge.automatic_confidence <= 1:
                raise AssemblyError("INVALID_INPUT", "edge confidences must be between 0 and 1")
            for candidate in edge.candidates:
                if not isinstance(candidate.measurement, (list, tuple)) or len(candidate.measurement) != 3:
                    raise AssemblyError("INVALID_INPUT", "candidate measurement must have three components")
                for value in (*candidate.measurement, candidate.candidate_score, candidate.intrinsic_cost,
                              candidate.slide, candidate.slide_cost, candidate.overlap_ratio, candidate.polygon_overlap_ratio):
                    finite_number(value, "candidate pose/score")
                indices = (candidate.wall_a_index, candidate.wall_b_index)
                if not all(isinstance(i, int) and not isinstance(i, bool) for i in indices) or not 0 <= indices[0] < len(converted[edge.room_a]) or not 0 <= indices[1] < len(converted[edge.room_b]):
                    raise AssemblyError("INVALID_INPUT", "candidate corner/wall indices do not match layout", pair_index=edge.index)
        if self.root_id is not None and self.root_id not in converted:
            raise AssemblyError("INVALID_INPUT", "root_id is not a room in the scene")

    def validate_state(self, state):
        unknown = set(state.placed) - set(self.polygons)
        if unknown:
            raise AssemblyError("UNKNOWN_ROOM", "state contains unknown rooms", room_ids=sorted(unknown))
        valid = {f"edge_{e.index:04d}" for e in self.edges}
        if set(state.excluded_pair_ids) - valid:
            raise AssemblyError("INVALID_INPUT", "excluded_pair_ids contains an unknown pair ID")
