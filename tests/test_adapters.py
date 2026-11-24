import copy
import json
import pytest

from room_assembly import Scene, AssemblyError, AssemblyState, recommend_rooms, suggest_placements
from room_assembly.adapters import build_scene, scene_to_dict, scene_from_dict, save_scene, load_scene
from room_assembly.adapters.legacy import read_da3_csv


def geo_result():
    return {"img_a":"A", "img_b":"B",
            "pose":{"R":[[1,0,0],[0,1,0],[0,0,1]], "cam_b_in_a":[-1,0,0], "az_ba_deg":90},
            "n":{"raw":10000,"inlier":4000},"inlier_rate":0.9,
            "quality":{"az_cov_a":0.5,"az_cov_b":0.5},"flags":{},
            "regions":[{"frac_of_inliers":1,"n":4000,"mean_conf":0.95,"a":{"area_frac":0.1},"b":{"area_frac":0.1}}]}


def test_guided_generation_without_candidate_csv(scene):
    generated = build_scene("generated", scene.polygons, geoverify_results=[geo_result()])
    assert len(generated.edges) == 1
    assert generated.edges[0].risk_tier == "low"
    assert generated.edges[0].manual_action != "layout_suggestion_pending"
    assert generated.metadata["unavailable_pairs"] == []
    assert generated.edges[0].measurement[:2] == pytest.approx((50,0))


def test_invalid_measurement_falls_back_to_da3(scene):
    geo = geo_result()
    geo["flags"]["no_covisibility"] = True
    generated = build_scene("s", scene.polygons, geoverify_results=[geo],
                             da3_pairs=[{"room_a":"A","room_b":"B","probability":0.9}])
    assert generated.metadata["unavailable_pairs"][0]["reason"] == "invalid_geoverify_measurement"
    assert generated.edges[0].manual_action == "layout_suggestion_pending"


def test_guided_pair_has_priority_over_da3(scene):
    generated = build_scene("s", scene.polygons, geoverify_results=[geo_result()],
                             da3_pairs=[{"room_a":"B","room_b":"A","probability":0.99}])
    assert len(generated.edges) == 1
    assert generated.edges[0].manual_action != "layout_suggestion_pending"


def test_invalid_measurement_without_fallback_has_no_connection(scene):
    geo = geo_result()
    geo["flags"]["degenerate_rot_only"] = True
    generated = build_scene("s", scene.polygons, geoverify_results=[geo])
    assert generated.edges == []
    assert recommend_rooms(generated, {"placed":{"A":{"x":0,"y":0,"theta_deg":0}}})["status"] == "no_available_connections"


def test_no_pairs_does_not_invent_connectivity(scene):
    generated = build_scene("s", scene.polygons)
    assert generated.edges == []


@pytest.mark.parametrize("probability", [-0.1,1.1,float("nan"),None,True])
def test_bad_da3_probability(scene, probability):
    with pytest.raises(AssemblyError):
        build_scene("s",scene.polygons,da3_pairs=[{"room_a":"A","room_b":"B","probability":probability}])


def test_below_da3_threshold_and_duplicate_reverse(scene):
    generated = build_scene("s",scene.polygons,da3_pairs=[
        {"room_a":"A","room_b":"C","probability":0.009},
        {"room_a":"A","room_b":"B","probability":0.9},
        {"room_a":"B","room_b":"A","probability":0.8}])
    assert len(generated.edges) == 1


def test_geoverify_duplicate_not_silently_selected(scene):
    a = geo_result()
    b = {**a,"img_a":"B","img_b":"A"}
    with pytest.raises(AssemblyError, match="duplicate"):
        build_scene("s",scene.polygons,geoverify_results=[a,b])


@pytest.mark.parametrize("points", [
    [[0,0],[1,1]], [[0,0],[1,1],[0,1],[1,0]],
    [[0,0],[1,0],[1,0],[0,1]], [[0,0],[1,0],[0,1],[0,0]],
    [[0,0],[1,0],[0,float("inf")]],
])
def test_invalid_polygon_rejected_without_reindexing(points):
    with pytest.raises(AssemblyError):
        Scene("s",{"A":points})


def test_scene_round_trip(scene, state, tmp_path):
    before = suggest_placements(scene,state,"A","B")
    destination = tmp_path / "scene.json"
    save_scene(scene, destination)
    loaded = load_scene(destination)
    assert suggest_placements(loaded,state,"A","B") == before
    assert scene_to_dict(loaded) == scene_to_dict(scene)


def test_scene_version_and_units(scene):
    data = scene_to_dict(scene)
    data["schema_version"] = "99"
    with pytest.raises(AssemblyError) as caught:
        scene_from_dict(data)
    assert caught.value.code == "UNSUPPORTED_SCHEMA"
    data["schema_version"] = "1.0"
    data["angle_unit"] = "degrees"
    with pytest.raises(AssemblyError):
        scene_from_dict(data)


def test_bad_corner_indices(scene):
    data = scene_to_dict(scene)
    data["pairs"][0]["candidates"][0]["wall_a_index"] = 999
    with pytest.raises(AssemblyError, match="indices"):
        scene_from_dict(data)


@pytest.mark.parametrize("change", ["short_pose","nan_score","string_wall","duplicate_index"])
def test_malformed_normalized_pair(scene, change):
    data = scene_to_dict(scene)
    if change == "short_pose":
        data["pairs"][0]["measurement"] = [0,0]
    elif change == "nan_score":
        data["pairs"][0]["confidence"] = float("nan")
    elif change == "string_wall":
        data["pairs"][0]["candidates"][0]["wall_a_index"] = "0"
    else:
        data["pairs"][1]["index"] = data["pairs"][0]["index"]
    with pytest.raises(AssemblyError):
        scene_from_dict(data)


def test_gt_pose_fields_in_geoverify_not_used(scene):
    a = geo_result()
    b = copy.deepcopy(a)
    b.update({"gt_pose":{"dx":99999,"dy":99999},"translation_error_gt":0,"gt_best_candidate_id":"invalid"})
    left = build_scene("s",scene.polygons,geoverify_results=[a])
    right = build_scene("s",scene.polygons,geoverify_results=[b])
    assert scene_to_dict(left) == scene_to_dict(right)


def test_da3_csv_ignores_label_and_only_parses_paths(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("pano_a_path,pano_b_path,probability,label\n/nonexistent/s/panos/A.jpg,/nonexistent/s/panos/B.jpg,0.9,0\n")
    assert read_da3_csv("s",path) == [{"room_a":"A","room_b":"B","probability":0.9}]


def test_bad_json(tmp_path):
    path = tmp_path/"scene.json"
    path.write_text('{"x": NaN}')
    with pytest.raises(AssemblyError) as caught:
        load_scene(path)
    assert caught.value.code == "INPUT_READ_ERROR"


def test_render_png_and_no_write_by_default(scene, state, tmp_path):
    from room_assembly import Pose2D, preview_placement
    from room_assembly.rendering import render_preview
    preview = preview_placement(scene,state,"B",Pose2D())
    data = render_preview(preview, width=320, height=240)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert list(tmp_path.iterdir()) == []
    destination = tmp_path/"preview.png"
    assert render_preview(preview, destination, width=320, height=240) == data
    assert destination.read_bytes() == data
