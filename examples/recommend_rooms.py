"""python examples/recommend_rooms.py --scene outputs/demo/scene.json --state outputs/demo/state.json"""
import argparse
import json
from pathlib import Path
from room_assembly import recommend_rooms
from room_assembly.adapters import load_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(recommend_rooms(load_scene(args.scene), json.loads(args.state.read_text()), args.top_k), indent=2))
