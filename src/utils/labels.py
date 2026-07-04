import json
from pathlib import Path
from typing import Dict, Optional


def labels_from_relation_json(relation_path: Path) -> Dict[str, str]:
    label_map = {}
    try:
        data = json.loads(relation_path.read_text(encoding="utf-8"))
        for p in data.get("panos", []):
            pid = p.get("id")
            name = p.get("name")
            if pid and name:
                label_map[pid] = name
    except Exception as e:
        print(f"[WARNING] Failed to parse relation.json: {e}")
    return label_map


def get_room_labels(scene_dir: Path, relation_json: Optional[Path] = None) -> Dict[str, str]:
    """
    Attempts to read data/raw/<house_id>/relation.json to map pano_ids to room names.
    <scene_dir> is e.g. data/group/58715_floor1
    The house_id is usually the first part of the folder name (e.g. 58715).
    """
    if relation_json is not None and relation_json.exists():
        return labels_from_relation_json(relation_json)

    scene_name = scene_dir.name
    house_id = scene_name.split("_")[0]

    relation_candidates = [
        # Typical relative structure from scene_dir: ../../raw/<house_id>/relation.json
        scene_dir.parent.parent / "raw" / house_id / "relation.json",
        Path.cwd()
        / "dataset"
        / "dollhouse_raw"
        / f"DollhouseTask_{house_id}_NoOutdoor"
        / f"DollhouseTask_{house_id}_NoOutdoor"
        / "relation.json",
    ]

    for relation_path in relation_candidates:
        if relation_path.exists():
            return labels_from_relation_json(relation_path)

    return {}


def get_display_label(pano_id: str, label_map: Dict[str, str]) -> str:
    """Returns the Chinese room name if available, otherwise the last 6 chars of the ID."""
    return label_map.get(pano_id, pano_id[-6:])
