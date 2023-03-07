"""Keys file to overcome TorchScript constants bug."""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

from nequip.data import register_fields

EDGE_ENERGY: Final[str] = "edge_energy"
EDGE_FORCES: Final[str] = "edge_forces"
EDGE_FEATURES: Final[str] = "edge_features"
CURL: Final[str] = "curl"


register_fields(edge_fields=[EDGE_ENERGY, EDGE_FEATURES])