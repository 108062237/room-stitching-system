import argparse
import matplotlib

matplotlib.use("Agg")
import json
import sys
import math
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.geom import load_layout_gt_txt_as_local_xy
from src.system.preview_pipeline import normalize_match_entries, transform_from_pairs


def compute_relative_pose(pA_start, pA_end, pB_start, pB_end):
    """計算 B 對齊到 A 的 dx, dy, dtheta（不加 + pi，與 verifier 一致）"""
    angA = math.atan2(pA_end[1] - pA_start[1], pA_end[0] - pA_start[0])
    angB = math.atan2(pB_end[1] - pB_start[1], pB_end[0] - pB_start[0])
    dtheta = angA - angB
    dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi  # 限制在 [-π, π]

    c, s = math.cos(dtheta), math.sin(dtheta)
    rx = c * pB_start[0] - s * pB_start[1]
    ry = s * pB_start[0] + c * pB_start[1]
    dx = pA_start[0] - rx
    dy = pA_start[1] - ry
    return dx, dy, dtheta


def main():
    parser = argparse.ArgumentParser(
        description="從完美的配對 JSON 產生 GTSAM edges.json"
    )
    parser.add_argument(
        "--matches", required=True, help="輸入的配對檔案 (例: perfect_matches.json)"
    )
    parser.add_argument(
        "--layout_dir",
        required=True,
        help="房間 txt 佈局檔所在的資料夾路徑 (例: layout_gt/)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="輸出的 edges JSON 檔案 (例: perfect_gtsam_edges.json)",
    )
    parser.add_argument(
        "--scene_id", required=False, help="場景 ID (不填則從 layout_dir 自動推斷)"
    )
    parser.add_argument(
        "--sigma_xy",
        type=float,
        default=0.005,
        help="極低的平移誤差標準差 (0.5cm) 作為強力縫合錨點",
    )
    parser.add_argument(
        "--sigma_theta",
        type=float,
        default=0.001,
        help="極低的旋轉誤差標準差 作為強力縫合錨點",
    )
    args = parser.parse_args()

    matches_json_path = args.matches
    base_dir = Path(args.layout_dir)
    out_edges_path = args.out
    scene_id = args.scene_id if args.scene_id else base_dir.parent.name

    matches_raw = json.load(open(matches_json_path, "r"))
    matches = normalize_match_entries(matches_raw)
    edges = []

    for edge in matches:
        src_id = edge["src"]
        dst_id = edge["dst"]
        
        poly_src = load_layout_gt_txt_as_local_xy(base_dir / edge["src_name"])
        poly_dst = load_layout_gt_txt_as_local_xy(base_dir / edge["dst_name"])

        if poly_src is None:
            raise ValueError(f"Cannot load layout: {base_dir / edge['src_name']}")
        if poly_dst is None:
            raise ValueError(f"Cannot load layout: {base_dir / edge['dst_name']}")

        pairs = edge.get("pairs", [])
        if len(pairs) < 2:
            print(f"⚠️ 略過配對 {src_id} -> {dst_id}，點數不足 2 對")
            continue

        # GTSAM BetweenFactorPose2(i, j, z_ij) uses z_ij = x_i^-1 * x_j.
        # As a transform matrix, this maps points from j-local coordinates into
        # i-local coordinates. Therefore src=i, dst=j needs dst_local -> src_local.
        h_dst_to_src = transform_from_pairs(poly_src, poly_dst, pairs)
        dx = float(h_dst_to_src[0, 2])
        dy = float(h_dst_to_src[1, 2])
        dtheta = math.atan2(h_dst_to_src[1, 0], h_dst_to_src[0, 0])

        # 完美匹配：使用非常小的雜訊（高置信度）
        edges.append(
            {
                "i": src_id,
                "j": dst_id,
                "measurement": {"dx": dx, "dy": dy, "dtheta": dtheta},
                "noise_sigma": {
                    "sigma_xy": args.sigma_xy,
                    "sigma_theta": args.sigma_theta,
                },
                "meta": {"source": "perfect_manual_match", "confidence": 1.0},
            }
        )

    output = {"scene_id": scene_id, "edges": edges}

    out_dir = Path(out_edges_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_edges_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"✅ 成功產生 {len(edges)} 條完美邊界: {out_edges_path}")


if __name__ == "__main__":
    main()
