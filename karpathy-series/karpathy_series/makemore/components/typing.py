from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import prod
from typing import Self

from torch import dtype, float16, float32


@dataclass(frozen=True)
class ScalarType:
    dtype: dtype


@dataclass(frozen=True)
class TokenType:
    size: int


@dataclass(frozen=True)
class Dims:
    value: int


@dataclass(frozen=True)
class Shape:
    value: list[Dims]

    def norm_index(self, dim: int) -> int:
        if not -len(self.value) <= dim < len(self.value):
            raise ValueError(f"Dimension {dim} out of bounds for shape {self}")

        return len(self.value) + dim if dim < 0 else dim

    def is_unit(self) -> bool:
        return len(self.value) == 0

    def active(self) -> Dims | None:
        return None if self.is_unit() else self.value[-1]

    def replace_active(self, x: Dims) -> Self:
        if len(self.value) == 0:
            return self
        return self.__class__([*self.value[:-1], x])

    def expand(self, x: Dims) -> Self:
        return self.__class__([*self.value, x])

    def flatten(self, dim: int) -> Self:
        dim = self.norm_index(dim)
        prefix, suffix = self.value[:dim], self.value[dim:]
        return self.__class__([*prefix, Dims(prod(d.value for d in suffix))])

    def extend(self, dim: int | None = None) -> Self:
        if dim is None or dim == len(self.value):
            return self.expand(Dims(1))

        dim = self.norm_index(dim)
        prefix, suffix = self.value[:dim], self.value[dim:]
        return self.__class__([*prefix, Dims(1), *suffix])

    def slide(self, dim: int, f: Dims) -> Self:
        dim = self.norm_index(dim)
        if dim == len(self.value) - 1:
            return self.factor(dim, f)

        prefix, site, followed, suffix = self.value[:dim], self.value[dim], self.value[dim + 1], self.value[dim + 2 :]
        q = site.value // f.value
        if site.value != q * f.value:
            raise ValueError(f"Factor {f} doesn't divide dimensions {site} at dimension {dim}")

        return self.__class__([*prefix, Dims(q), Dims(f.value * followed.value), *suffix])

    def factor(self, dim: int, f: Dims) -> Self:
        return self.extend(dim + 1 if dim != -1 else None).slide(dim, f)

    @classmethod
    def from_ints(cls, vs: Iterable[int]) -> Self:
        return cls([Dims(k) for k in vs])


@dataclass(frozen=True)
class ArrayType:
    shape: Shape

    def replace_active(self, x: Dims) -> Self:
        return replace(self, shape=self.shape.replace_active(x))


@dataclass(frozen=True)
class TensorType(ArrayType):
    data: ScalarType


@dataclass(frozen=True)
class TokenArrayType(ArrayType):
    tokens: TokenType

    def embed(self, scalar: ScalarType, embedding_dims: Dims | None = None) -> TensorType:
        if embedding_dims is None:
            embedded = Dims(self.tokens.size)
        else:
            embedded = embedding_dims
        return TensorType(self.shape.expand(embedded), scalar)


class ArrayTypeError(Exception):
    pass


def Float16(*vs: int) -> TensorType:
    return TensorType(Shape.from_ints(vs), ScalarType(float16))


def Float32(*vs: int) -> TensorType:
    return TensorType(Shape.from_ints(vs), ScalarType(float32))


def Tokens(count: int, *vs: int) -> TokenArrayType:
    return TokenArrayType(Shape.from_ints(vs), TokenType(count))
