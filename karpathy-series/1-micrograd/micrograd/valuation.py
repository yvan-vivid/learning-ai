from dataclasses import dataclass


@dataclass
class Valuation:
    value: float = 0
    gradient: float = 0

    def __str__(self) -> str:
        return f"value = {self.value}, grad = {self.gradient}"
