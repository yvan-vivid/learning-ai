import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from math import exp
from operator import mul
from typing import ClassVar, Sequence, TypeAlias, override

from .valuation import Valuation


@dataclass(frozen=True)
class ValueTypeBase:
    pass


@dataclass(frozen=True)
class Variable(ValueTypeBase):
    pass


@dataclass(frozen=True)
class Operator(ValueTypeBase, ABC):
    glyph: ClassVar[str]

    @abstractmethod
    def forward(self, operands: Sequence[Valuation]) -> Valuation: ...

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
    glyph: ClassVar[str] = "+"
    bias: float = 0

    @override
    def __str__(self) -> str:
        return str(self.glyph) if self.bias == 0 else f"{self.glyph} {self.bias}"

    @override
    def forward(self, operands: Sequence[Valuation]) -> Valuation:
        return Valuation(sum(v.value for v in operands) + self.bias)

    @override
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
    glyph: ClassVar[str] = "×"
    coefficient: float = 1

    @override
    def __str__(self) -> str:
        return str(self.glyph) if self.coefficient == 1 else f"{self.coefficient}{self.glyph}"

    @override
    def forward(self, operands: Sequence[Valuation]) -> Valuation:
        return Valuation(reduce(lambda x, y: x * y, (v.value for v in operands)) * self.coefficient)

    @override
    def backward(self, result: Valuation, operands: Sequence[Valuation]) -> None:
        """
        For the product
            f(x_1, ..., x_k, c) = c Prod k: n . x_k
        the derivative is
            D[x_k]f = c * (Prod j: n . x_j) / x_k
                where x_k != 0
        when x_k is 0, there is not an easy out aside from taking the product of the other values
        """
        for op in operands:
            if op.value != 0:
                op.gradient += result.gradient * result.value / op.value
            else:
                op.gradient += result.gradient * reduce(mul, (op2.value for op2 in operands if op != op2), 1)


@dataclass(frozen=True)
class Pow(Operator):
    glyph: ClassVar[str] = "^"
    exponent: float

    @override
    def __str__(self) -> str:
        return f"{self.glyph}{self.exponent}"

    @override
    def forward(self, operands: Sequence[Valuation]) -> Valuation:
        assert len(operands) == 1
        operand = operands[0]
        return Valuation(math.pow(operand.value, self.exponent))

    @override
    def backward(self, result: Valuation, operands: Sequence[Valuation]) -> None:
        """
        For the power
            x^q
        the derivative is, match q
            D[x]x^q = q x^(q-1)
        """
        if self.exponent == 0:
            return

        assert len(operands) == 1
        operand = operands[0]
        operand.gradient += result.gradient * self.exponent * operand.value ** (self.exponent - 1)


@dataclass(frozen=True)
class Tanh(Operator):
    glyph: ClassVar[str] = "tanh"

    @override
    def __str__(self) -> str:
        return str(self.glyph)

    @override
    def forward(self, operands: Sequence[Valuation]) -> Valuation:
        assert len(operands) == 1
        operand = operands[0]
        p = exp(2 * operand.value)
        return Valuation((p - 1) / (p + 1))

    @override
    def backward(self, result: Valuation, operands: Sequence[Valuation]) -> None:
        assert len(operands) == 1
        operand = operands[0]
        operand.gradient += result.gradient * (1 - result.value**2)


@dataclass(frozen=True)
class Exp(Operator):
    glyph: ClassVar[str] = "exp"

    @override
    def __str__(self) -> str:
        return str(self.glyph)

    @override
    def forward(self, operands: Sequence[Valuation]) -> Valuation:
        assert len(operands) == 1
        operand = operands[0]
        return Valuation(exp(operand.value))

    @override
    def backward(self, result: Valuation, operands: Sequence[Valuation]) -> None:
        assert len(operands) == 1
        operand = operands[0]
        operand.gradient += result.gradient * result.value


ValueType: TypeAlias = Variable | Operator
