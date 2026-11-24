"""Public stateless interfaces. All returned values are JSON serializable."""
from __future__ import annotations

import math
from numbers import Integral
from .core.geometry import compose
from .core.placement import PlacementEngine
from .core.recommendation import rank_starting_rooms, find_available_connections, rank_connections
from .core.preview import transform_preview_geometry
from .errors import AssemblyError
from .models import Scene, AssemblyState, Pose2D


def _context(scene, current_state):
    if not isinstance(scene, Scene):
        raise AssemblyError("INVALID_INPUT", "scene must be a Scene; use a scene adapter")
    state = AssemblyState.parse(current_state)
    scene.validate_state(state)
    return state


def _limit(top_k):
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
        raise AssemblyError("INVALID_INPUT", "top_k must be a positive integer")
    return int(top_k)


def _id(edge):
    return f"edge_{edge.index:04d}"


def recommend_rooms(scene, current_state, top_k=5):
    """Rank available insertion connections, or suggest starting rooms.

    This is a next-step ranking, not a fixed full-floor insertion sequence.
    Different connections to the same unplaced room remain separate entries.
    """
    state = _context(scene, current_state)
    limit = _limit(top_k)
    if not state.placed:
        ranked, scores = rank_starting_rooms(scene)
        return {"scene_id": scene.scene_id, "mode": "start", "status": "available",
                "score_direction": "higher_is_better", "score_type": "weighted_degree",
                "total": len(ranked), "items": [
                    {"rank": i+1, "room_id": room, "connect_to": None, "pair_id": None,
                     "score": float(scores[room])} for i, room in enumerate(ranked[:limit])]}
    choices = rank_connections(find_available_connections(scene.edges, state.placed, state.excluded_pair_ids))
    remaining = len(set(scene.polygons) - set(state.placed))
    return {"scene_id": scene.scene_id, "mode": "insertion",
            "status": "available" if choices else "all_rooms_placed" if not remaining else "no_available_connections",
            "remaining_rooms": remaining, "score_direction": "higher_is_better",
            "score_type": "connection_confidence", "total": len(choices),
            "items": [{"rank": i+1, **item} for i, item in enumerate(choices[:limit])]}


def suggest_placements(scene, current_state, room_a, room_b, top_k=5, *, pair_id=None):
    """Return relative B-in-A poses and absolute B poses, without committing.

    A must be placed and B unplaced. Internal ranking/snap calculations retain
    the placement algorithm; this API does not expose a conflict-check service.
    """
    state = _context(scene, current_state)
    limit = _limit(top_k)
    for room in (room_a, room_b):
        if room not in scene.polygons:
            raise AssemblyError("UNKNOWN_ROOM", "room is not in scene", room_id=room)
    if room_a not in state.placed or room_b in state.placed or room_a == room_b:
        raise AssemblyError("INVALID_STATE", "room_a must be placed and room_b must be unplaced")
    pairs = [edge for edge in scene.edges if {edge.room_a, edge.room_b} == {room_a, room_b}
             and (pair_id is None or _id(edge) == pair_id)]
    if not pairs:
        raise AssemblyError("NO_PAIR_DATA", "no position candidates for the requested pair")
    if len(pairs) > 1 and pair_id is None:
        raise AssemblyError("AMBIGUOUS_PAIR", "specify pair_id returned by recommend_rooms")
    edge = pairs[0]
    engine = PlacementEngine(scene.polygons, {k: pose.radians() for k, pose in state.placed.items()})
    options = engine._layout_options(edge, room_a, room_b)
    parent_pose = state.placed[room_a].radians()
    items = []
    for i, option in enumerate(options[:limit]):
        dx, dy, theta = option["measurement"]
        x, y, absolute = compose(parent_pose, (dx, dy, theta))
        items.append({"rank": i+1, "candidate_id": option["candidate_id"],
                      "relative_pose": {"dx": float(dx), "dy": float(dy), "dtheta_deg": math.degrees(theta)},
                      "world_pose": {"x": float(x), "y": float(y), "theta_deg": math.degrees(absolute)},
                      "score": float(option["layout_score"])})
    return {"scene_id": scene.scene_id, "pair_id": _id(edge), "room_a": room_a, "room_b": room_b,
            "score_direction": "lower_is_better", "score_type": "layout_cost", "total": len(options), "items": items}


def preview_placement(scene, current_state, room_id, world_pose):
    """Transform outlines only. Never snap, check conflicts, optimize or save.

    May also preview an already placed room at a replacement pose. All other
    room poses are unchanged. Identical input always gives identical geometry.
    """
    state = _context(scene, current_state)
    if room_id not in scene.polygons:
        raise AssemblyError("UNKNOWN_ROOM", "room is not in scene", room_id=room_id)
    pose = Pose2D.parse(world_pose)
    poses = {**state.placed, room_id: pose}
    rooms, bounds = transform_preview_geometry(scene.polygons, poses, room_id)
    return {"scene_id": scene.scene_id, "coordinate_frame": "world_xy", "units": "layout_units",
            "preview_room_id": room_id, "rooms": rooms,
            "bounds": bounds}
