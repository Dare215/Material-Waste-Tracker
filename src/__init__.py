# Ensure proper imports for Streamlit Cloud
import sys
from pathlib import Path

root = Path(__file__).resolve().parent  # /src
repo = root.parent                      # /Material-Waste-Tracker

for p in {str(root), str(repo)}:
    if p not in sys.path:
        sys.path.insert(0, p)
