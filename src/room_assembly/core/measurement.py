"""Room assembly calculations."""
from __future__ import annotations

import math
from typing import Any
CAMERA_TO_ROOM_LOCAL_CONVENTION = "room-stitching local [x,y] = [-camera_x, -camera_z]"

def normalize_angle_deg(angle: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def angle_error_deg(a: float, b: float) -> float:
    return abs(normalize_angle_deg(float(a) - float(b)))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unit2(x: float, y: float) -> list[float] | None:
    norm = math.hypot(float(x), float(y))
    if norm < 1e-12:
        return None
    return [float(x) / norm, float(y) / norm]


def _vector_angle_deg(a: list[float], b: list[float]) -> float:
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    return math.degrees(math.acos(dot))


def _bearing_from_azimuth(az_deg: float) -> list[float]:
    """Convert GeoVerify azimuth to [camera_x, camera_z].

    GeoVerify azimuth is 0 deg at image center / forward direction, positive
    toward +x.  Thus x=sin(az), z=cos(az).
    """
    a = math.radians(float(az_deg))
    return [math.sin(a), math.cos(a)]


def _bearing_angle_deg(v: list[float] | None) -> float | None:
    if v is None:
        return None
    return normalize_angle_deg(math.degrees(math.atan2(v[0], v[1])))


def geoverify_horizontal_to_room_local(bearing_xz: list[float] | None) -> list[float] | None:
    """Convert GeoVerify horizontal [camera_x, camera_z] to room local [x, y].

    room-stitching-system's layout_gt projection maps ERP center/camera +z to
    room local -Y, and camera +x to room local -X.  Because rectify_polygon()
    rotates back to the original frame, this fixed -I conversion is stable.
    """
    if bearing_xz is None:
        return None
    return _unit2(-float(bearing_xz[0]), -float(bearing_xz[1]))


def extract_relative_yaw(R: list[list[float]]) -> float | None:
    """Extract room-local B-in-A horizontal yaw from GeoVerify pose.R.

    GeoVerify R maps A-frame directions into B-frame directions.  For a room
    measurement A -> B, we need B local axes expressed in A frame, so use R.T.
    Project R.T to the horizontal x/z plane:

        x_A = R.T @ x_B

    The first B-local horizontal axis expressed in A is [R00, R02] in [x,z],
    giving yaw atan2(z_component, x_component).  This matches the existing
    pose-edge prototype's convention, but this script only reports yaw.

    The camera->room horizontal transform is fixed M=-I.  For 2D rotations,
    R_room = M * R_camera * M^-1 = R_camera, so no extra 180-degree yaw offset
    is applied.
    """
    if not (isinstance(R, list) and len(R) >= 3):
        return None
    try:
        r00 = float(R[0][0])
        r02 = float(R[0][2])
    except (TypeError, ValueError, IndexError):
        return None
    return normalize_angle_deg(math.degrees(math.atan2(r02, r00)))


def project_horizontal_bearing(vector3: Any) -> list[float] | None:
    """Project a 3D vector to a unit [camera_x, camera_z] horizontal bearing."""
    if not (isinstance(vector3, list) and len(vector3) >= 3):
        return None
    x = _as_float(vector3[0])
    z = _as_float(vector3[2])
    if x is None or z is None:
        return None
    return _unit2(x, z)


def extract_bearing_ab(pose: dict[str, Any]) -> tuple[float | None, list[float] | None, str | None]:
    """Extract A -> B bearing in A camera frame."""
    bearing = project_horizontal_bearing(pose.get("cam_b_in_a"))
    if bearing is not None:
        return _bearing_angle_deg(bearing), bearing, "cam_b_in_a_xz"
    az = _as_float(pose.get("az_ab_deg"))
    if az is None:
        return None, None, None
    return normalize_angle_deg(az), _bearing_from_azimuth(az), "az_ab_deg"


def extract_bearing_ba(pose: dict[str, Any]) -> tuple[float | None, list[float] | None, str | None]:
    """Extract B -> A bearing in B camera frame."""
    az = _as_float(pose.get("az_ba_deg"))
    if az is None:
        return None, None, None
    return normalize_angle_deg(az), _bearing_from_azimuth(az), "az_ba_deg"


def _transform_bearing_b_to_a(R: list[list[float]], bearing_b: list[float]) -> list[float] | None:
    """Transform a B-frame horizontal [x,z] vector into A frame using R.T."""
    try:
        x_b, z_b = float(bearing_b[0]), float(bearing_b[1])
        r00 = float(R[0][0])
        r20 = float(R[2][0])
        r02 = float(R[0][2])
        r22 = float(R[2][2])
    except (TypeError, ValueError, IndexError):
        return None
    # v_a = R.T @ [x_b, 0, z_b]; keep horizontal [x,z].
    return _unit2(r00 * x_b + r20 * z_b, r02 * x_b + r22 * z_b)


def compute_bearing_consistency(
    R: list[list[float]] | None,
    bearing_ab: list[float] | None,
    bearing_ba: list[float] | None,
) -> float | None:
    """Compare A->B with the opposite of B->A transformed into A frame."""
    if R is None or bearing_ab is None or bearing_ba is None:
        return None
    ba_in_a = _transform_bearing_b_to_a(R, bearing_ba)
    if ba_in_a is None:
        return None
    expected_ab = [-ba_in_a[0], -ba_in_a[1]]
    return _vector_angle_deg(bearing_ab, expected_ab)


def _short(v: float | None) -> float | None:
    return None if v is None else round(float(v), 6)


def build_room_measurement(
    res: dict[str, Any],
    room_a: str,
    room_b: str,
    gt_edge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pose = res.get("pose") if isinstance(res.get("pose"), dict) else {}
    flags = res.get("flags") if isinstance(res.get("flags"), dict) else {}
    quality = res.get("quality") if isinstance(res.get("quality"), dict) else {}

    no_covisibility = bool(flags.get("no_covisibility"))
    degenerate_rot_only = bool(flags.get("degenerate_rot_only"))
    measurement_valid = not no_covisibility
    rotation_valid = measurement_valid and bool(pose.get("R"))
    bearing_valid = measurement_valid and not degenerate_rot_only

    R = pose.get("R")
    relative_yaw = extract_relative_yaw(R) if rotation_valid else None
    camera_bearing_ab_deg, camera_bearing_ab, bearing_ab_source = extract_bearing_ab(pose) if bearing_valid else (None, None, None)
    camera_bearing_ba_deg, camera_bearing_ba, bearing_ba_source = extract_bearing_ba(pose) if bearing_valid else (None, None, None)
    room_bearing_ab = geoverify_horizontal_to_room_local(camera_bearing_ab)
    room_bearing_ba = geoverify_horizontal_to_room_local(camera_bearing_ba)
    room_bearing_ab_deg = _bearing_angle_deg(room_bearing_ab)
    room_bearing_ba_deg = _bearing_angle_deg(room_bearing_ba)
    consistency = compute_bearing_consistency(R, camera_bearing_ab, camera_bearing_ba) if bearing_valid else None

    out = {
        "room_a": room_a,
        "room_b": room_b,
        "img_a": res.get("img_a"),
        "img_b": res.get("img_b"),
        "rotation_convention": "GeoVerify pose.R maps A frame to B frame: x_b = R * x_a + t",
        "yaw_extraction": "room-local B-in-A yaw derived from GeoVerify camera R.T projected to the horizontal x/z plane; no extra 180 deg because camera->room is fixed -I",
        "camera_to_room_local": CAMERA_TO_ROOM_LOCAL_CONVENTION,
        "relative_yaw_deg": _short(relative_yaw),
        "camera_bearing_ab_deg": _short(camera_bearing_ab_deg),
        "camera_bearing_ab": None if camera_bearing_ab is None else [_short(camera_bearing_ab[0]), _short(camera_bearing_ab[1])],
        "room_bearing_ab_deg": _short(room_bearing_ab_deg),
        "room_bearing_ab": None if room_bearing_ab is None else [_short(room_bearing_ab[0]), _short(room_bearing_ab[1])],
        "bearing_ab_source": bearing_ab_source,
        "camera_bearing_ba_deg": _short(camera_bearing_ba_deg),
        "camera_bearing_ba": None if camera_bearing_ba is None else [_short(camera_bearing_ba[0]), _short(camera_bearing_ba[1])],
        "room_bearing_ba_deg": _short(room_bearing_ba_deg),
        "room_bearing_ba": None if room_bearing_ba is None else [_short(room_bearing_ba[0]), _short(room_bearing_ba[1])],
        "bearing_ba_source": bearing_ba_source,
        "internal_bearing_consistency_error_deg": _short(consistency),
        "inlier_rate": res.get("inlier_rate"),
        "conf_used": res.get("conf_used"),
        "quality": quality,
        "rotation_valid": bool(rotation_valid),
        "bearing_valid": bool(bearing_valid and room_bearing_ab is not None and room_bearing_ba is not None),
        "measurement_valid": bool(measurement_valid),
        "flags": {
            "no_covisibility": no_covisibility,
            "degenerate_rot_only": degenerate_rot_only,
        },
    }

    if gt_edge:
        m = gt_edge.get("measurement") or {}
        gt_yaw = _as_float(m.get("dtheta"))
        gt_x = _as_float(m.get("dx"))
        gt_y = _as_float(m.get("dy"))
        if gt_yaw is not None:
            gt_yaw_deg = normalize_angle_deg(math.degrees(gt_yaw))
            out["gt_yaw_deg"] = _short(gt_yaw_deg)
            out["yaw_error_deg"] = None if relative_yaw is None else _short(angle_error_deg(relative_yaw, gt_yaw_deg))
        if gt_x is not None and gt_y is not None:
            gt_bearing = _unit2(gt_x, gt_y)
            if gt_bearing is not None:
                out["gt_bearing_ab"] = [_short(gt_bearing[0]), _short(gt_bearing[1])]
                out["bearing_error_deg"] = None if room_bearing_ab is None else _short(_vector_angle_deg(room_bearing_ab, gt_bearing))
                out["old_camera_bearing_error_deg"] = None if camera_bearing_ab is None else _short(_vector_angle_deg(camera_bearing_ab, gt_bearing))
    return out
