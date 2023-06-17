from dataclasses import dataclass
from typing import Dict
from typing_extensions import Self

from micrograd.value_type import Operator
from micrograd.assignment import Assignment
from micrograd.valuation import Valuation


@dataclass(frozen=True)
class GraphValuation:
    assignment: Assignment
    assigned: Dict[int, Valuation]

    def forward(self) -> None:
        for node in self.assignment.graph.topological():
            model = node.data
            if isinstance(model, Operator):
                self.assigned[node.ident] = model.forward(
                    tuple(self.assigned[p.ident] for p in node.pred)
                )

    def backward(self) -> None:
        graph = self.assignment.graph
        for node in self.assignment.graph.topological(reverse=True):
            model = node.data
            if graph.is_root(node):
                self.assigned[node.ident].gradient = 1
            if isinstance(model, Operator):
                model.backward(
                    self.assigned[node.ident],
                    tuple(self.assigned[p.ident] for p in node.pred),
                )

    @classmethod
    def initialize(cls, assignment: Assignment) -> Self:
        return cls(
            assignment,
            {
                node.ident: Valuation(assignment.assigned.get(node.ident, 0))
                for node in assignment.graph.nodes()
            },
        )

    @classmethod
    def run(cls, assignment: Assignment) -> Self:
        gv = cls.initialize(assignment)
        gv.forward()
        gv.backward()
        return gv
