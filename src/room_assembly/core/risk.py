"""Room assembly calculations."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from .measurement import compute_bearing_consistency, extract_bearing_ab, extract_bearing_ba

RISK_SIGNALS = {
    "few_inliers_128": lambda row: float(row["n_inlier"]) <= 128.0,
    "low_inlier_rate_018": lambda row: float(row["inlier_rate"]) <= 0.18,
    "tiny_region_area_000325": lambda row: float(row["region_area_min_sum"]) <= 0.00325,
    "bearing_inconsistent_033": lambda row: float(row["bearing_consistency_deg"]) >= 0.33,
    "few_raw_matches_3827": lambda row: float(row["n_raw"]) <= 3827.0,
    "low_azimuth_coverage_00695": lambda row: float(row["az_cov_min"]) <= 0.0695,
    "low_region_confidence_0758": lambda row: (
        parse_float(row.get("region_mean_conf")) is not None
        and float(row["region_mean_conf"]) <= 0.758
    ),
}


def parse_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def region_metrics(regions: list[dict[str, Any]], n_inlier: int) -> dict[str, float | int | None]:
    fractions = [
        value for region in regions
        if (value := parse_float(region.get("frac_of_inliers"))) is not None
    ]
    fractions.sort(reverse=True)
    hhi = sum(value * value for value in fractions)
    weighted_conf_num = 0.0
    weighted_conf_den = 0.0
    area_a = 0.0
    area_b = 0.0
    for region in regions:
        count = parse_float(region.get("n")) or 0.0
        confidence = parse_float(region.get("mean_conf"))
        if confidence is not None and count > 0.0:
            weighted_conf_num += count * confidence
            weighted_conf_den += count
        area_a += parse_float((region.get("a") or {}).get("area_frac")) or 0.0
        area_b += parse_float((region.get("b") or {}).get("area_frac")) or 0.0
    return {
        "region_count": len(regions),
        "region_top_fraction": fractions[0] if fractions else None,
        "region_top2_fraction": sum(fractions[:2]) if fractions else None,
        "region_hhi": hhi if fractions else None,
        "region_effective_count": 1.0 / hhi if hhi > 0.0 else None,
        "region_mean_conf": safe_divide(weighted_conf_num, weighted_conf_den),
        "region_area_min_sum": min(area_a, area_b),
        "region_inlier_coverage": safe_divide(weighted_conf_den, float(n_inlier)),
    }


def extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    counts = data.get("n") or {}
    quality = data.get("quality") or {}
    flags = data.get("flags") or {}
    pose = data.get("pose") or {}
    n_raw = int(counts.get("raw", 0) or 0)
    n_conf = int(counts.get("conf", 0) or 0)
    n_used = int(counts.get("used", 0) or 0)
    n_inlier = int(counts.get("inlier", 0) or 0)
    lat_drop = int(counts.get("lat_drop", 0) or 0)
    az_cov_a = parse_float(quality.get("az_cov_a"))
    az_cov_b = parse_float(quality.get("az_cov_b"))
    wedge_a = parse_float(quality.get("az_wedge_a_deg"))
    wedge_b = parse_float(quality.get("az_wedge_b_deg"))
    rotation = pose.get("R")
    _, bearing_ab, _ = extract_bearing_ab(pose)
    _, bearing_ba, _ = extract_bearing_ba(pose)
    bearing_consistency = compute_bearing_consistency(rotation, bearing_ab, bearing_ba)
    result = {
        "n_raw": n_raw,
        "n_conf": n_conf,
        "n_used": n_used,
        "n_inlier": n_inlier,
        "conf_keep_rate": safe_divide(float(n_conf), float(n_raw)),
        "inlier_rate": parse_float(data.get("inlier_rate")),
        "lat_drop_rate": safe_divide(float(lat_drop), float(n_conf + lat_drop)),
        "sampson_med_deg": parse_float(quality.get("sampson_med_deg")),
        "sampson_p90_deg": parse_float(quality.get("sampson_p90_deg")),
        "rot_only_med_deg": parse_float(quality.get("rot_only_med_deg")),
        "az_cov_min": min(az_cov_a, az_cov_b) if az_cov_a is not None and az_cov_b is not None else None,
        "az_cov_mean": 0.5 * (az_cov_a + az_cov_b) if az_cov_a is not None and az_cov_b is not None else None,
        "az_wedge_min_deg": min(wedge_a, wedge_b) if wedge_a is not None and wedge_b is not None else None,
        "az_wedge_mean_deg": 0.5 * (wedge_a + wedge_b) if wedge_a is not None and wedge_b is not None else None,
        "bearing_consistency_deg": bearing_consistency,
        "baseline_txz": parse_float(pose.get("baseline_txz")),
        "flag_low_matches": bool(flags.get("low_matches")),
        "flag_no_covisibility": bool(flags.get("no_covisibility")),
        "flag_degenerate_rot_only": bool(flags.get("degenerate_rot_only")),
        "flag_doorway_wedge": bool(flags.get("doorway_wedge")),
    }
    result.update(region_metrics(data.get("regions") or [], n_inlier))
    return result
