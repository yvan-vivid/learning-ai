from functools import reduce
from typing import Callable, Tuple

from .assignment import Assignment
from .value import Value, ValueDag, ValueGraph


class Neuron:
    name: str
    graph: ValueDag
    weights: Tuple[Value, ...]
    bias: Value
    output: Value

    def __init__(self, name: str, graph: ValueGraph, inputs: Tuple[Value, ...]) -> None:
        assert len(inputs) != 0
        self.name = name
        self.graph = inputs[0].graph
        self.weights = tuple(graph(f"{name}.w{k}") for k, _ in enumerate(inputs))
        self.bias = graph(f"{name}.bias")
        linear = graph.sum(self.bias, *((w * x) for w, x in zip(self.weights, inputs))) | f"{name}.linear"
        self.output = linear.tanh() | name

    def assign(self, *weights: float) -> Assignment:
        w = tuple(weights)
        assert len(w) == len(self.weights)
        return Assignment.create(self.graph, dict(zip(self.weights, weights)) | {self.bias: 0})

    def assign_from(self, weighting: Callable[[], float]) -> Assignment:
        return self.assign(*(weighting() for _ in range(len(self.weights))))


class Layer:
    name: str
    graph: ValueDag
    neurons: Tuple[Neuron, ...]
    outputs: Tuple[Value, ...]

    def __init__(self, name: str, graph: ValueGraph, n: int, inputs: Tuple[Value, ...]) -> None:
        assert len(inputs) != 0
        self.name = name
        self.graph = inputs[0].graph
        self.neurons = tuple(Neuron(f"{name}.n{k}", graph, inputs) for k in range(n))
        self.outputs = tuple(neuron.output for neuron in self.neurons)

    def assign_from(self, weighting: Callable[[], float]) -> Assignment:
        return reduce(lambda a, b: a | b, (neuron.assign_from(weighting) for neuron in self.neurons))
