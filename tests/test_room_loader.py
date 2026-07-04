from pathlib import Path

import numpy as np
import pytest

SCENE_DIR = Path("data/group/58472_Floor1/layout_gt")


@pytest.mark.skipif(not SCENE_DIR.exists(), reason="local example scene not available")
def test_layout_file_exists():
    assert SCENE_DIR.exists()


def test_numpy_available():
    arr = np.array([[1, 2], [3, 4]])
    assert arr.shape == (2, 2)
