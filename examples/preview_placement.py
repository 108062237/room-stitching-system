"""Preview a supplied world pose; overlapping placements are also rendered."""
import argparse
import json
from pathlib import Path
from room_assembly import preview_placement
from room_assembly.adapters import load_scene
from room_assembly.rendering import render_preview

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--room", required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--theta-deg", type=float, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/preview.png"))
    args = parser.parse_args()
    result = preview_placement(load_scene(args.scene), json.loads(args.state.read_text()), args.room,
                               {"x": args.x, "y": args.y, "theta_deg": args.theta_deg})
    render_preview(result, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.output.resolve())
