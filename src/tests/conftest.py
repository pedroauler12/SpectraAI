import sys
from pathlib import Path

# Add repo root to sys.path for tests
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
