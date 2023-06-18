from dataclasses import dataclass
from typing import Dict
from typing_extensions import Self

from micrograd.value import ValueDag, ValueGraph, Value
from micrograd.value_type import Variable


@dataclass(frozen=True)
class Assignment:
    graph: ValueDag
    assigned: Dict[int, float]

    def is_complete(self) -> bool:
        return frozenset(self.assigned.keys()) == frozenset(
            n.ident for n in self.graph.entries())

    def __or__(self: Self, other: Self) -> Self:
        assert self.graph == other.graph
        return self.__class__(self.graph, self.assigned | other.assigned)

    @classmethod
    def create(cls, graph_like: ValueDag | ValueGraph, assign: Dict[Value, float]) -> Self:
        graph = graph_like.graph if isinstance(graph_like, ValueGraph) else graph_like
        assigned = {}
        for value_node, value in assign.items():
            assert value_node.graph == graph
            assert value_node.node.data == Variable()
            assigned[value_node.node.ident] = value
        return cls(graph, assigned)
