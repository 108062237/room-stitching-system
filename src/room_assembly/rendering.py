"""Optional image rendering. Does not set matplotlib's global backend."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

PALETTE = ("#277da1", "#43aa8b", "#f9c74f", "#f9844a", "#577590", "#90be6d", "#c05a8e", "#4d908e", "#bc6c25")


def render_preview(preview, path=None, *, width=1000, height=700, dpi=120, show_labels=True):
    """Return PNG bytes; write a file only if an explicit destination is given."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Polygon

    figure = Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    for room in preview["rooms"]:
        color = PALETTE[int.from_bytes(sha256(room["room_id"].encode()).digest()[:4], "big") % len(PALETTE)]
        axis.add_patch(Polygon(room["polygon"], facecolor=color, edgecolor=color, alpha=0.4,
                               linewidth=3 if room["preview"] else 1.8,
                               linestyle="--" if room["preview"] else "-"))
        if show_labels:
            pose = room["world_pose"]
            axis.text(pose["x"], pose["y"], room["room_id"][-6:], ha="center", va="center", fontsize=9)
    bounds = preview["bounds"]
    span = max(bounds["max_x"]-bounds["min_x"], bounds["max_y"]-bounds["min_y"], 1.0)
    padding = span * 0.08
    axis.set_xlim(bounds["min_x"]-padding, bounds["max_x"]+padding)
    axis.set_ylim(bounds["min_y"]-padding, bounds["max_y"]+padding)
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.tight_layout(pad=0.2)
    output = BytesIO()
    canvas.print_png(output)
    data = output.getvalue()
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return data
