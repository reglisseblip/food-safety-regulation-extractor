"""conftest — puts src/ and benchmark/ on sys.path (same convention as benchmark/run.py)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
