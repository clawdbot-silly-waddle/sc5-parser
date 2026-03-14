"""Generated FlatBuffer schemas - add this directory to sys.path for imports."""
import sys
from pathlib import Path

_schemas_dir = str(Path(__file__).parent)
if _schemas_dir not in sys.path:
    sys.path.insert(0, _schemas_dir)