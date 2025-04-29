from typing import Optional, Set

from nova.src.backend.autodiff import Node


def print_graph(node: Node, level: int = 0) -> None:
    indent = "  " * level
    # Determine a label for this node:
    if node.operation and node.operation.name:
        label = node.operation.name
    elif node.role:
        label = node.role
    else:
        label = "Leaf"

    # If there are parent nodes, build a string that shows each parent's index and its role.
    if node.parents:
        operands = " & ".join(
            f"{parent.idx} [{' ' + parent.role if parent.role else 'Leaf'}]".strip()
            for parent in node.parents
        )
        print(f"{indent}{label} (node_idx={node.idx}) ({operands})")
    else:
        print(f"{indent}{label} (node_idx={node.idx})")

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

    name = node.operation.__class__.__name__ if node.operation else "Leaf"
    grad_str = f", grad={node.grad}" if node.grad is not None else ""
    print(f"{indent}{name} (value={node.value}{grad_str})")

    for parent in node.parents:
        print_graph_with_grad(parent, level + 1, visited)


def print_tree(graph, node, parent=None, prefix="", is_last=True):
    if parent is None:
        print(node)
        children = graph[node]
        for i, child in enumerate(children):
            child_is_last = i == len(children) - 1
            print_tree(graph, child, node, "", child_is_last)
    else:
        connector = "└──> " if is_last else "├──> "
        print(prefix + connector + str(node))

        new_prefix = prefix + ("    " if is_last else "│   ")
        children = [child for child in graph[node] if child != parent]
        for i, child in enumerate(children):
            child_is_last = i == len(children) - 1
            print_tree(graph, child, node, new_prefix, child_is_last)
