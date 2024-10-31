from dataclasses import dataclass
from typing import override


@dataclass
class Valuation:
    value: float = 0
    gradient: float = 0

    @override
    def __str__(self) -> str:
        return f"value = {self.value}, grad = {self.gradient}"
