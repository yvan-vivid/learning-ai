from abc import abstractmethod
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import TypeAlias, Sequence, ClassVar

from micrograd.valuation import Valuation


@dataclass(frozen=True)
class ValueTypeBase:
    pass


@dataclass(frozen=True)
class Variable(ValueTypeBase):
    pass


@dataclass(frozen=True)
class Operator(ValueTypeBase):
    glyph: ClassVar[str]

    @abstractmethod
    def forward(self, operands: Sequence[Valuation]) -> Valuation:
        ...

    @abstractmethod
    def backward(self, result: Valuation, operands: Sequence[Valuation]) -> None:
        """
        The differential of any operator over n variables is
            df(x_1, ..., x_n) = Sum k: n . D[x_k]f dx_k
        For some variable V, and assuming x_k are functions of variables, the
        derivative is thus
            D[V]f = Sum k: n . D[x_k]f D[V]x_k
        hence we want to back prop
            head * D[x_k]f
        to each node k
        """
        ...


@dataclass(frozen=True)
class Sum(Operator):
    glyph = "+"
    bias: float = 0

    def __str__(self) -> str:
        return str(self.glyph) if self.bias == 0 else f"{self.glyph} {self.bias}"

    def forward(self, operands: Sequence[Valuation]) -> Valuation:
        return Valuation(sum(v.value for v in operands) + self.bias)

    def backward(self, result: Valuation, operands: Sequence[Valuation]) -> None:
        """
        For the sum
            f(x_1, ..., x_k, b) = b + Sum k: n . x_k
        the derivative is
            D[x_k]f = 1
        """
        for op in operands:
            op.gradient += result.gradient


@dataclass(frozen=True)
class Prod(Operator):
    glyph = "×"
    coefficient: float = 1

    def __str__(self) -> str:
        return str(self.glyph) if self.coefficient == 1 else f"{self.coefficient}{self.glyph}"

    def forward(self, operands: Sequence[Valuation]) -> Valuation:
        return Valuation(reduce(mul, (v.value for v in operands)) * self.coefficient)

    def backward(self, result: Valuation, operands: Sequence[Valuation]) -> None:
        """
        For the product
            f(x_1, ..., x_k, c) = c Prod k: n . x_k
        the derivative is
            D[x_k]f = c * (Prod j: n . x_j) / x_k
        """
        for op in operands:
            op.gradient += result.gradient * result.value / op.value


ValueType: TypeAlias = Variable | Operator
