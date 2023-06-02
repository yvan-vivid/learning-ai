from dataclasses import dataclass
from typing import Dict
from typing_extensions import Self

from micrograd.value import ValueGraph, Value
from micrograd.value_type import Variable


@dataclass(frozen=True)
class Assignment:
    graph: ValueGraph
    assigned: Dict[int, float]

    def is_complete(self) -> bool:
        return frozenset(self.assigned.keys()) == frozenset(
            n.ident for n in self.graph.graph.entries())

    @classmethod
    def create(cls, graph: ValueGraph, assign: Dict[Value, float]) -> Self:
        assigned = {}
        for value_node, value in assign.items():
            assert value_node.graph == graph.graph
            assert value_node.node.data == Variable()
            assigned[value_node.node.ident] = value
        return cls(graph, assigned)
