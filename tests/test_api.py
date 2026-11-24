import copy
import json
import math
import builtins
import pytest
import numpy as np

from room_assembly import AssemblyError, AssemblyState, Pose2D, recommend_rooms, suggest_placements, preview_placement
from room_assembly.core.geometry import compose, invert


def test_start_matches_reference_root_policy(scene):
    result = recommend_rooms(scene, AssemblyState())
    assert result["mode"] == "start"
    assert result["items"][0]["room_id"] == "A"  # DA3 fallback excluded from root degree.
    assert all(i["score"] == 0 for i in result["items"])


def test_insertion_ranking_and_update(scene, state):
    result = recommend_rooms(scene, state)
    assert [(i["room_id"], i["connect_to"]) for i in result["items"]] == [("B", "A"), ("C", "A")]
    assert [i["score"] for i in result["items"]] == pytest.approx([0.225, 0.15])
    result = recommend_rooms(scene, AssemblyState({"A": Pose2D(), "B": Pose2D(60, 0)}))
    assert [(i["room_id"], i["connect_to"]) for i in result["items"]] == [("C", "B"), ("C", "A")]


def test_excluded_connections(scene, state):
    state.excluded_pair_ids = ("edge_0000",)
    assert [i["room_id"] for i in recommend_rooms(scene, state)["items"]] == ["C"]


def test_no_candidate_is_not_complete(scene):
    result = recommend_rooms(scene, AssemblyState({"A":Pose2D(), "B":Pose2D(), "C":Pose2D()}))
    assert result["status"] == "no_available_connections"
    assert result["remaining_rooms"] == 1
    assert result["items"] == []


def test_all_placed(scene):
    result = recommend_rooms(scene, AssemblyState({r: Pose2D() for r in scene.polygons}))
    assert result["status"] == "all_rooms_placed"
    assert result["items"] == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "2"])
def test_top_k_validation(scene, state, top_k):
    with pytest.raises(AssemblyError, match="top_k"):
        recommend_rooms(scene, state, top_k)
    with pytest.raises(AssemblyError, match="top_k"):
        suggest_placements(scene, state, "A", "B", top_k)


def test_relative_world_composition(scene, state):
    result = suggest_placements(scene, state, "A", "B", 100)
    assert len(result["items"]) == result["total"]
    assert [i["score"] for i in result["items"]] == sorted(i["score"] for i in result["items"])
    for item in result["items"]:
        relative = item["relative_pose"]
        expected = compose(state.placed["A"].radians(), (relative["dx"], relative["dy"], math.radians(relative["dtheta_deg"])))
        assert Pose2D.parse(item["world_pose"]).radians() == pytest.approx(expected)


def test_reversed_pair_direction(scene):
    state = AssemblyState({"B": Pose2D(-20, 50, 30)})
    result = suggest_placements(scene, state, "B", "A", 1)
    assert result["room_a"] == "B" and result["room_b"] == "A"
    pose = result["items"][0]["world_pose"]
    relative = result["items"][0]["relative_pose"]
    assert Pose2D.parse(pose).radians() == pytest.approx(compose(state.placed["B"].radians(),
        (relative["dx"],relative["dy"],math.radians(relative["dtheta_deg"]))))


@pytest.mark.parametrize("a,b,code", [("A","D","NO_PAIR_DATA"),("B","A","INVALID_STATE"),("A","missing","UNKNOWN_ROOM")])
def test_unavailable_placement(scene, state, a, b, code):
    with pytest.raises(AssemblyError) as caught:
        suggest_placements(scene, state, a, b)
    assert caught.value.code == code
    assert caught.value.to_dict()["message"]


def test_preview_exact_transform_and_no_conflict_gate(scene):
    state = AssemblyState({"A":Pose2D()})
    preview = preview_placement(scene, state, "B", Pose2D(0, 0, 90))
    points = next(i["polygon"] for i in preview["rooms"] if i["room_id"] == "B")
    np.testing.assert_allclose(points, [[15,-20],[15,20],[-15,20],[-15,-20]], atol=1e-10)
    # This B lies fully inside A. No geometry success/rejection is returned.
    assert set(preview) == {"scene_id","coordinate_frame","units","preview_room_id","rooms","bounds"}
    assert preview["bounds"] == {"min_x":-30.0,"min_y":-20.0,"max_x":30.0,"max_y":20.0}


def test_preview_does_not_call_suggestion_or_checks(scene, state, monkeypatch):
    from room_assembly.core.placement import PlacementEngine
    def fail(*args, **kwargs):
        raise AssertionError("preview must not run snap, scoring or checks")
    for method in ("_layout_options", "_snap_world_pose", "_placement_metrics"):
        monkeypatch.setattr(PlacementEngine, method, fail)
    assert preview_placement(scene, state, "B", Pose2D())["preview_room_id"] == "B"


def test_preview_existing_room_preserves_others(scene):
    state = AssemblyState({"A":Pose2D(), "B":Pose2D(20,30,40)})
    preview = preview_placement(scene, state, "A", Pose2D(10,20,50))
    b = next(i for i in preview["rooms"] if i["room_id"] == "B")
    assert b["world_pose"] == state.placed["B"].to_dict()
    assert not b["preview"]


def test_no_input_mutation_and_json_outputs(scene, state):
    from room_assembly.adapters import scene_to_dict
    before_scene = json.dumps(scene_to_dict(scene), sort_keys=True)
    before_state = copy.deepcopy(state.to_dict())
    outputs = [recommend_rooms(scene, state), suggest_placements(scene, state, "A", "B"),
               preview_placement(scene, state, "B", Pose2D())]
    for output in outputs:
        json.dumps(output, allow_nan=False)
    assert json.dumps(scene_to_dict(scene), sort_keys=True) == before_scene
    assert state.to_dict() == before_state


def test_public_calls_do_not_read_or_write_files(scene, state, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("public calculation attempted file I/O")
    monkeypatch.setattr(builtins, "open", fail)
    assert recommend_rooms(scene, state)["items"]
    assert suggest_placements(scene, state, "A", "B")["items"]
    assert preview_placement(scene, state, "B", Pose2D())["rooms"]


@pytest.mark.parametrize("value", [float("nan"),float("inf"),"5",True,None])
def test_bad_pose_rejected(value):
    with pytest.raises(AssemblyError):
        Pose2D(value, 0, 0)


def test_unknown_state_room(scene):
    with pytest.raises(AssemblyError) as caught:
        recommend_rooms(scene, {"placed":{"missing":{"x":0,"y":0,"theta_deg":0}}})
    assert caught.value.code == "UNKNOWN_ROOM"


def test_pose_inverse():
    pose = (24,-35,1.7)
    assert compose(pose,invert(pose)) == pytest.approx((0,0,0),abs=1e-10)
