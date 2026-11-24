"""Import existing candidate CSVs. No original web checkout is needed."""
import argparse
from pathlib import Path
from room_assembly.adapters import load_legacy_scene, save_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--combined-results", type=Path, required=True)
    parser.add_argument("--candidates-csv", type=Path)
    parser.add_argument("--geoverify-root", type=Path)
    parser.add_argument("--da3-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    save_scene(load_legacy_scene(**args), output)
    print(output.resolve())
