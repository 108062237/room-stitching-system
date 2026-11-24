"""Generate candidates from layouts + model outputs, without legacy CSVs/GT poses."""
import argparse
from pathlib import Path
from room_assembly.adapters import build_scene, save_scene
from room_assembly.adapters.legacy import read_layout_directory, read_geoverify_directory, read_da3_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--geoverify-root", type=Path)
    parser.add_argument("--da3-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.geoverify_root and not args.da3_predictions:
        parser.error("provide GeoVerify and/or DA3 results")
    scene = build_scene(args.scene_id, read_layout_directory(args.layout_dir),
                        geoverify_results=read_geoverify_directory(args.scene_id, args.geoverify_root) if args.geoverify_root else [],
                        da3_pairs=read_da3_csv(args.scene_id, args.da3_predictions) if args.da3_predictions else [])
    save_scene(scene, args.output)
    print(args.output.resolve())
