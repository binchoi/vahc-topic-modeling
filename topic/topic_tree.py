
from collections import defaultdict


class TopicTree:
    child_to_parent = {}
    parent_to_children = {}

    def __init__(self):
        self.child_to_parent = {}
        self.parent_to_children = defaultdict(list)

    def _add_edge(self, child: str, parent: str):
        self.child_to_parent[child] = parent
        self.parent_to_children[parent].append(child)

    def add_edges(self, edges: list[tuple[str, str]]):
        for child, parent in edges:
            self._add_edge(child, parent)

    def get_ancestors(self, child: str) -> list[str]:
        ancestors = []
        while child in self.child_to_parent:
            child = self.child_to_parent[child]
            ancestors.append(child)
        return ancestors

    def get_siblings_inclusive(self, node: str):
        if node not in self.child_to_parent:
            return [node]
        return self.parent_to_children[self.child_to_parent[node]]