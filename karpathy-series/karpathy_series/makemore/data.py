from collections.abc import Iterator
from pathlib import Path


def read_data(path: Path) -> list[str]:
    with path.open("r") as handle:
        return handle.read().splitlines()


def read_character_data(path: Path) -> str:
    return path.read_text()


def read_blocked_character_data(path: Path, block_size: int) -> Iterator[str]:
    with path.open("r") as handle:
        while len(out := handle.read(block_size)) > 0:
            yield out
