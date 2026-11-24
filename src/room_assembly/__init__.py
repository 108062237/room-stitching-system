"""Three stateless public calculations; input adapters are separate."""
from .api import recommend_rooms, suggest_placements, preview_placement
from .models import Scene, AssemblyState, Pose2D
from .errors import AssemblyError

__version__ = "0.1.1"
__all__ = ["recommend_rooms", "suggest_placements", "preview_placement", "Scene", "AssemblyState", "Pose2D", "AssemblyError"]

