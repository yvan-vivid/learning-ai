from abc import ABCMeta
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, override


# Manages a labeled collection of identifiers
@dataclass
class IdentManager:
    ident_source: int = 0
    label_table: Dict[int, str] = field(default_factory=dict)

    def use(self) -> int:
        new_ident = self.ident_source
        self.ident_source += 1
        return new_ident

    def set_label(self, item: int, label: str) -> None:
        self.label_table[item] = label

    def label(self, ident: int) -> Optional[str]:
        return self.label_table.get(ident)

    def labels(self) -> FrozenSet[str]:
        return frozenset(self.label_table.values())

    @override
    def __str__(self) -> str:
        return f"IdentManager({self.ident_source})"

    @override
    def __hash__(self) -> int:
        return id(self)


class HasIdentities(metaclass=ABCMeta):
    identities: IdentManager
