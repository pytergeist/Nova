from typing import Set, Optional

from abditus.autodiff._node import Node


def print_graph(node: Node, level: int = 0) -> None:
    indent = "  " * level
    op_name = node.operation.__class__.__name__ if node.operation else "Leaf"
    print(f"{indent}{op_name} (value={node.value})")
    for parent in node.parents:
        print_graph(parent, level + 1)


def print_graph_with_grad(
    node: None, level: int = 0, visited: Optional[Set[int]] = None
) -> None:
    if visited is None:
        visited = set()
    indent = "  " * level
    node_id = id(node)

    if node_id in visited:
        print(f"{indent}... (already printed)")
        return
    visited.add(node_id)

    op_name = node.operation.__class__.__name__ if node.operation else "Leaf"
    grad_str = f", grad={node.grad}" if node.grad is not None else ""
    print(f"{indent}{op_name} (value={node.value}{grad_str})")

    for parent in node.parents:
        print_graph_with_grad(parent, level + 1, visited)
