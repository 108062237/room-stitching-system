"""Import existing research artifacts without constructing a web Session."""
from __future__ import annotations

import csv
from pathlib import Path

from .candidate_csv import load_edges, load_base_candidates
from .layout import load_layout_gt_txt_as_local_xy
from .scene import _json_read, _args, append_da3_edges, choose_root
from ..core.candidates import enrich_edges_from_data
from ..errors import AssemblyError
from ..models import Scene


def read_layout_directory(layout_dir, *, pano_width=1024, pano_height=512, layout_z=50.0):
    layout_dir = Path(layout_dir)
    paths = sorted(layout_dir.glob("*.txt"))
    if not paths:
        raise AssemblyError("INPUT_READ_ERROR", "no layout TXT files", path=str(layout_dir))
    polygons = {}
    for path in paths:
        try:
            polygon = load_layout_gt_txt_as_local_xy(path, pano_width, pano_height, layout_z)
        except (ValueError, OSError) as exc:
            raise AssemblyError("INVALID_INPUT", "cannot load layout TXT", path=str(path)) from exc
        if polygon is None:
            raise AssemblyError("INVALID_INPUT", "layout has too few floor corners", path=str(path))
        polygons[path.stem] = polygon
    return polygons


def read_geoverify_directory(scene_id, root):
    """root is the directory containing <scene_id>/pair_results/*.json."""
    results = []
    if not Path(root).is_dir():
        raise AssemblyError("INPUT_READ_ERROR", "GeoVerify root is not a directory", path=str(root))
    for path in sorted((Path(root) / scene_id / "pair_results").glob("*.json")):
        results.append(_json_read(path))
    return results


def read_da3_csv(scene_id, path):
    """Adapt the current pano_a_path/pano_b_path CSV to explicit room IDs."""
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"pano_a_path", "pano_b_path", "probability"} <= set(reader.fieldnames or []):
                raise AssemblyError("INVALID_INPUT", "DA3 CSV missing required columns")
            pairs = []
            for row in reader:
                a = Path(row["pano_a_path"])
                b = Path(row["pano_b_path"])
                if a.parent.name != "panos" or a.parent.parent.name != scene_id:
                    continue
                if b.parent.name != "panos" or b.parent.parent.name != scene_id:
                    raise AssemblyError("INVALID_INPUT", "DA3 pair crosses scene boundaries")
                pairs.append({"room_a": a.stem, "room_b": b.stem, "probability": float(row["probability"])})
            return pairs
    except (OSError, ValueError) as exc:
        if isinstance(exc, AssemblyError):
            raise
        raise AssemblyError("INPUT_READ_ERROR", "cannot read DA3 CSV", path=str(path)) from exc


def load_legacy_scene(scene_id, *, layout_dir, combined_results, candidates_csv=None,
                      geoverify_root=None, da3_predictions=None):
    """Use existing candidate artifacts for exact reference-comparison runs.

    All paths are explicit; no sibling checkout, user state or manifest needed.
    GT pose/error columns in legacy CSVs are not used by candidate selection.
    """
    polygons = read_layout_directory(layout_dir)
    # Validate polygons before candidate expansion touches corner indices.
    Scene(scene_id, polygons)
    try:
        edges = [edge for edge in load_edges(Path(combined_results)).get(scene_id, [])
                 if edge.room_a in polygons and edge.room_b in polygons]
        grouped = load_base_candidates(Path(candidates_csv)) if candidates_csv is not None else {}
        if candidates_csv is not None and not Path(candidates_csv).is_file():
            raise FileNotFoundError(candidates_csv)
        for (a, b), candidates in grouped.items():
            if a in polygons and b in polygons:
                for c in candidates:
                    if not (0 <= c.wall_a_index < len(polygons[a]) and 0 <= c.wall_b_index < len(polygons[b])):
                        raise AssemblyError("INVALID_INPUT", "candidate indices do not match layout", room_a=a, room_b=b)
        index = {}
        if geoverify_root is not None:
            for raw in read_geoverify_directory(scene_id, geoverify_root):
                a, b = raw.get("img_a"), raw.get("img_b")
                if a and b:
                    index[(scene_id, tuple(sorted((a, b))))] = (Path("geoverify.json"), raw)
        enrich_edges_from_data(scene_id, edges, polygons, grouped, index, _args())
        root = choose_root(polygons, edges)
        if da3_predictions is not None:
            append_da3_edges(edges, polygons, read_da3_csv(scene_id, da3_predictions))
        return Scene(scene_id, polygons, edges, root,
                     {"candidate_origin": "legacy_csv", "layout_provenance": "legacy_layout_gt",
                      "pair_selection_provenance": "existing_batch_pairs_may_be_hotspot_selected"})
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        if isinstance(exc, AssemblyError):
            raise
        raise AssemblyError("INPUT_READ_ERROR", "cannot import legacy candidates", reason=str(exc)) from exc
