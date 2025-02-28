from pathlib import Path


def read_data(path: Path) -> list[str]:
    with path.open("r") as handle:
        return handle.read().splitlines()
