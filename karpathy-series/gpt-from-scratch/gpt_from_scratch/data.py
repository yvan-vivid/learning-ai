"""Data"""
from pathlib import Path

def load_shakespear(repo: Path) -> str:
    """Load the tiny shakespear dataset"""
    with open(repo / "tiny-shakespear.txt", "r", encoding="utf8") as handle:
        return handle.read()
