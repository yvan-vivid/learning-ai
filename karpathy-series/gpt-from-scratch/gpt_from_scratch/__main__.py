"""Main"""
from pathlib import Path
from gpt_from_scratch.encoding import CharacterEncoding
from gpt_from_scratch.data import load_shakespear

from torch import Tensor, tensor, long as torch_long


def load_shakespear_sequence(repo: Path) -> Tensor:
    """Load shakespear into a tensor"""
    shakespear = load_shakespear(repo)
    encoder = CharacterEncoding.from_character_set(set(shakespear))
    encoded = encoder.encode(shakespear)
    return tensor(encoded, dtype=torch_long)


def setup() -> None:
    """Mirroring repl"""
    repo = Path("../data")
    data = load_shakespear(repo)
    print(len(data))

    split_n = int(0.9*len(data))
    training = data[:split_n]
    validation = data[split_n:]

    print(len(training), len(validation))



setup()
