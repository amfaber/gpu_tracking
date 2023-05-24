from .gpu_tracking import *
from .lib import *

__all__ = ["batch", "characterize_points", "link", "connect", "LoG", "load", "mean_from_file"]

import sys

def run_app():
    from pathlib import Path
    index_path = Path(__file__).parent / "docs" / "_build" / "html" / "index.html"
    if len(sys.argv) >= 2 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        import webbrowser
        webbrowser.open(index_path)
        sys.exit(0)
    tracking_app(index_path)