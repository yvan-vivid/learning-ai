from dataclasses import dataclass, field
from typing import Optional, TypeAlias, Tuple, Iterable
from typing_extensions import Self

from micrograd.graph import Dag, DagNode
from micrograd.value_type import ValueType, Variable, Sum, Prod, Tanh, Exp, Pow


ValueNode: TypeAlias = DagNode[ValueType]
ValueDag: TypeAlias = Dag[ValueType]


@dataclass(frozen=True)
class Value:
    graph: ValueDag
    node: ValueNode

    @property
    def value_type(self) -> ValueType:
        return self.node.data

    def label(self) -> Optional[str]:
        return self.node.label()

    def node_like(self: Self, value: ValueType, *pred: ValueNode) -> Self:
        return self.__class__(self.graph, self.graph.node(value, *pred))

    def __or__(self: Self, label: str) -> Self:
        self.node.set_label(label)
        return self

    # Immediate
    def __add__(self: Self, other: Self | float) -> Self:
        if isinstance(other, (float, int)):
            return self.node_like(Sum(other), self.node)
        return self.node_like(Sum(), self.node, other.node)

    def __mul__(self: Self, other: Self | float) -> Self:
        if isinstance(other, (float, int)):
            return self.node_like(Prod(other), self.node)
        return self.node_like(Prod(), self.node, other.node)

    def tanh(self: Self) -> Self:
        return self.node_like(Tanh(), self.node)
    
    def exp(self: Self) -> Self:
        return self.node_like(Exp(), self.node)
    
    def __pow__(self: Self, exponent: int | float) -> Self:
        return self.node_like(Pow(exponent), self.node)
    # Derived
    def __radd__(self: Self, other: Self | float) -> Self:
        return self.__add__(other)

    def __rmul__(self: Self, other: Self | float) -> Self:
        return self * other

    def __neg__(self: Self) -> Self:
        return -1*self
    
    def __sub__(self: Self, other: Self | float) -> Self:
        return self + -other
    
    def __truediv__(self: Self, other: Self | float) -> Self:
        if isinstance(other, (float, int)):
            return self * (1/other)
        return self * other ** -1

    def __str__(self) -> str:
        return f"Value({self.node.data!s})"

    def __hash__(self) -> int:
        return hash(self.node)


@dataclass(frozen=True)
class ValueGraph:
    graph: ValueDag = field(default_factory=Dag)

    def __call__(self, label: Optional[str] = None) -> Value:
        return Value(self.graph, self.graph.node(Variable(), label=label))

    def __getitem__(self, labels: Tuple[str, ...]) -> Iterable[Value]:
        return (self(label) for label in labels)

    def sum(self, *values: Value) -> Value:
        return Value(self.graph, self.graph.node(Sum(), *(value.node for value in values))) 

