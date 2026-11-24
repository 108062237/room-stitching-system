"""Request B-in-A position suggestions, in descending preference."""
import argparse
import json
from pathlib import Path
from room_assembly import suggest_placements
from room_assembly.adapters import load_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--room-a", required=True)
    parser.add_argument("--room-b", required=True)
    parser.add_argument("--pair-id")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    result = suggest_placements(load_scene(args.scene), json.loads(args.state.read_text()),
                                args.room_a, args.room_b, args.top_k, pair_id=args.pair_id)
    print(json.dumps(result, indent=2))
