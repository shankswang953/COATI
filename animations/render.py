"""Render the standalone COATI Manim film without importing the training package."""
import argparse
import ctypes
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(root / "animations/media/mpl-cache"))
# Some mixed-architecture macOS installations leave pycairo symbols unbound.
# Preload an existing native ARM library only when the normal import fails.
try:
    import cairo
except ImportError:
    if sys.platform != "darwin":
        raise
    lib = Path("/opt/homebrew/lib/libcairo.2.dylib")
    if not lib.exists():
        raise
    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    import cairo
from manim import tempconfig
from coati_intro import COATIIntro

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    with tempconfig({"quality": "medium_quality" if args.preview else "high_quality",
                     "frame_rate": 24, "media_dir": str(root / "animations/media"),
                     "output_file": "COATIIntro"}):
        COATIIntro().render()
