from dataclasses import dataclass, field
from typing import Generic, Iterable, Optional, Set, Tuple, TypeVar, override

from typing_extensions import Self

from .ident import HasIdentities, IdentManager

D = TypeVar("D")


# This should not be constructed directly, but rather through a Dag which manages node identity
# In order for the node to remain frozen, the `data` field should be a reference or static
@dataclass(frozen=True)
class DagNode(Generic[D]):
    ident: int
    graph: HasIdentities
    data: D

    # This has to be a tuple since operands can be used several times
    #     i.e. `y = x + x`
    # Here, we can assume `x` is a reference to the same node.
    pred: Tuple["DagNode[D]", ...] = ()

    def set_label(self: Self, label: str) -> Self:
        self.graph.identities.set_label(self.ident, label)
        return self

    def label(self) -> Optional[str]:
        return self.graph.identities.label(self.ident)

    def pred_values(self) -> Tuple[D, ...]:
        return tuple(p.data for p in self.pred)

    @override
    def __str__(self) -> str:
        data_str = str(self.data)
        return f"Node({self.ident, data_str})"


# This is a forumlation of a Dag that can be a Dag by construction
# Because the DagNode type has a fixed tuple of predecessors, these
# must be defined earlier, hence a topological order. This can be defeated
# by assigning to `.pred`, but hay.
@dataclass
class Dag(Generic[D], HasIdentities):
    node_table: dict[int, DagNode[D]] = field(default_factory=dict)
    label_table: dict[int, str] = field(default_factory=dict)
    identities: IdentManager = field(default_factory=IdentManager)

    # As the graph is built, keep track of roots
    _roots: Set[int] = field(default_factory=set)
    _entries: Set[int] = field(default_factory=set)

    def node(self, data: D, *pred: DagNode[D], label: Optional[str] = None) -> DagNode[D]:
        new_node_ident = self.identities.use()
        new_node = DagNode(new_node_ident, self, data, tuple(pred))
        self.node_table[new_node_ident] = new_node

        if label is not None:
            _ = new_node.set_label(label)

        # New node is an entry when it has no predecessors
        if len(pred) == 0:
            self._entries.add(new_node_ident)

        # New node is definitely a root because it has yet to be referenced
        self._roots.add(new_node.ident)

        # Predecessors can now no longer be roots
        self._roots -= set(p.ident for p in pred)
        return new_node

    def is_root(self, node: DagNode[D]) -> bool:
        return node.ident in self._roots

    def roots(self) -> Iterable[DagNode[D]]:
        return (self.node_table[n] for n in self._roots)

    def entries(self) -> Iterable[DagNode[D]]:
        return (self.node_table[n] for n in self._entries)

    def nodes(self) -> Iterable[DagNode[D]]:
        return self.node_table.values()

    @override
    def __hash__(self) -> int:
        return hash(self.identities)

    # This extremely simple topological ordering relies on the idea that
    # the graph is constructed in a topological order (unless pred is
    # modified in nodes).
    def topological(self, reverse: bool = False) -> Iterable[DagNode[D]]:
        forward = self.node_table.values()
        return reversed(forward) if reverse else forward
