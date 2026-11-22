"""Independent, explicitly scoped SSS-1 reimplementation."""
import os
from pathlib import Path

# Keep optional libraries' caches inside this independent workspace.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent/"outputs/.matplotlib"))

__version__ = "0.1.0"
