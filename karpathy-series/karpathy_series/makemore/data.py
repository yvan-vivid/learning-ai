from pathlib import Path
from typing import List


def read_data(path: Path) -> List[str]:
    with path.open("r") as handle:
        return handle.read().splitlines()
