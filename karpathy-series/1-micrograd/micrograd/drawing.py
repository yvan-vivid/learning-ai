from typing import Literal, TypeAlias, List, Tuple, TypeVar

from graphviz import Digraph

from micrograd.graph import Dag, DagNode
from micrograd.value import ValueGraph
from micrograd.value_type import Variable, Operator
from micrograd.valuation import Valuation
from micrograd.calculation import GraphValuation

DrawFormat: TypeAlias = Literal["png", "svg"]
RankDir: TypeAlias = Literal["TB", "LR"]

D = TypeVar("D")
NodeList: TypeAlias = List[DagNode[D]]
EdgeList: TypeAlias = List[Tuple[DagNode[D], DagNode[D]]]


class Drawing:
    rankdir: RankDir = "LR"
    format: DrawFormat = "svg"

    def digraph(self) -> Digraph:
        return Digraph(format=self.format, graph_attr={"rankdir": self.rankdir})

    @staticmethod
    def _nodes_and_edges(dag: Dag[D]) -> Tuple[NodeList[D], EdgeList[D]]:
        nodes, edges = [], []
        for node in dag.node_table.values():
            nodes.append(node)
            for pred in node.pred:
                edges.append((pred, node))

        return nodes, edges

    def draw_underlying(self, dag: Dag[D]) -> Digraph:
        dot = self.digraph()
        nodes, edges = self._nodes_and_edges(dag)
        for node in nodes:
            node_ident = str(node.ident)
            dot.node(name=node_ident, label=node.label() or node_ident, shape="circle")

        for n1, n2 in edges:
            dot.edge(str(n1.ident), str(n2.ident))

        return dot

    @staticmethod
    def _op_ident(ident) -> str:
        return f"{ident}_op"

    def draw_value_graph(self, graph: ValueGraph) -> Digraph:
        dot = self.digraph()
        nodes, edges = self._nodes_and_edges(graph.graph)
        for node in nodes:
            node_ident = str(node.ident)
            label = node.label() or node_ident
            match node.data:
                case Variable():
                    dot.node(name=node_ident, label=label, shape="square")
                case Operator() as op:
                    op_ident = self._op_ident(node_ident)
                    dot.node(name=node_ident, label=label, shape="record")
                    dot.node(name=op_ident, label=str(op), shape="circle")
                    dot.edge(op_ident, node_ident)

        for n1, n2 in edges:
            dot.edge(str(n1.ident), self._op_ident(str(n2.ident)))

        return dot

    def draw_graph_valuation(self, valuation: GraphValuation) -> Digraph:
        dot = self.digraph()
        nodes, edges = self._nodes_and_edges(valuation.assignment.graph.graph)
        for node in nodes:
            node_ident = str(node.ident)
            node_val = valuation.assigned.get(node.ident, Valuation())
            label = f"{node.label() or node_ident} | {node_val!s}"
            match node.data:
                case Variable():
                    dot.node(name=node_ident, label=label, shape="record")
                case Operator() as op:
                    op_ident = self._op_ident(node_ident)
                    dot.node(name=node_ident, label=label, shape="record")
                    dot.node(name=op_ident, label=str(op), shape="circle")
                    dot.edge(op_ident, node_ident)

        for n1, n2 in edges:
            dot.edge(str(n1.ident), self._op_ident(str(n2.ident)))

        return dot

    # @staticmethod
    # def _op_id(node: DagNode[ValueData]) -> str:
    #     return str(node.ident) + node.data.operation
    #
    # def draw_values(self, values: ValueGraph) -> Digraph:
    #             name = node.label() or node_ident
    #             label, shape = f"'{name}' | data = {node.data.value} | grad = {node.data.gradient}", "record"
    #             if op != "input":
    #                 op_ident = self._op_id(node)
    #                 dot.node(name=op_ident, label=op)
    #                 dot.edge(op_ident, node_ident)
    #
    #         dot.node(name=node_ident, label=label, shape=shape)
    #
    #     for n1, n2 in edges:
    #         dot.edge(self._node_id(n1), self._op_id(n2))
    #
    #     return dot
