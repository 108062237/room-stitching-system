"""Run the three public calls on synthetic data; no private data or GPU needed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from room_assembly import AssemblyState, Pose2D, recommend_rooms, suggest_placements, preview_placement
from room_assembly.adapters import build_scene, load_scene, save_scene
from room_assembly.rendering import render_preview


def sample_scene():
    payload = json.loads((Path(__file__).parent / "data/input.json").read_text(encoding="utf-8"))
    return build_scene(**payload)


def run(scene, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = recommend_rooms(scene, AssemblyState())
    root = start["items"][0]["room_id"]
    state = AssemblyState({root: Pose2D()})
    recommendations = recommend_rooms(scene, state)
    if not recommendations["items"]:
        raise RuntimeError("example needs at least one available insertion connection")
    recommendation = recommendations["items"][0]
    placements = suggest_placements(scene, state, recommendation["connect_to"], recommendation["room_id"],
                                     pair_id=recommendation["pair_id"])
    preview = preview_placement(scene, state, recommendation["room_id"], placements["items"][0]["world_pose"])
    for name, data in (("start", start), ("state", state.to_dict()), ("recommendations", recommendations),
                       ("placements", placements), ("preview", preview)):
        (output / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    render_preview(preview, output / "preview.png")
    save_scene(scene, output / "scene.json")
    print(json.dumps({"output": str(output.resolve()), "placed_root": root,
                      "recommended_room": recommendation["room_id"], "preview": str(output / "preview.png")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, help="optional normalized scene JSON; default is synthetic data")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    run(load_scene(args.scene) if args.scene else sample_scene(), args.output)
