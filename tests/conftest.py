import pytest
from room_assembly.adapters import build_scene
from room_assembly import AssemblyState, Pose2D


@pytest.fixture
def scene():
    return build_scene("test_scene", {
        "A": [[-30,-20],[30,-20],[30,20],[-30,20]],
        "B": [[-20,-15],[20,-15],[20,15],[-20,15]],
        "C": [[-10,-10],[10,-10],[10,10],[-10,10]],
        "D": [[-10,-15],[10,-15],[10,15],[-10,15]],
    }, da3_pairs=[{"room_a":"A", "room_b":"B", "probability":0.9},
                  {"room_a":"A", "room_b":"C", "probability":0.6},
                  {"room_a":"B", "room_b":"C", "probability":0.8}])


@pytest.fixture
def state():
    return AssemblyState({"A": Pose2D(100, 200, 90)})
