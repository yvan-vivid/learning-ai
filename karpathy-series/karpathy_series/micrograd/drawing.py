from typing import List, Literal, Tuple, TypeAlias, TypeVar

from graphviz import Digraph

from .calculation import GraphValuation
from .graph import Dag, DagNode
from .valuation import Valuation
from .value import ValueGraph
from .value_type import Operator, Variable

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
        nodes: NodeList[D] = []
        edges: EdgeList[D] = []
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
    def _op_ident(ident: str) -> str:
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
        nodes, edges = self._nodes_and_edges(valuation.assignment.graph)
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
