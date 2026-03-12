"""
agent/langgraph_compat.py

Compatibility shim for LangGraph.

If ``langgraph`` is installed, this module re-exports its graph primitives.
If not installed, it provides a small in-process fallback that supports the
subset used by this project and test suite:
  - ``StateGraph``
  - ``START`` / ``END`` sentinels
  - ``compile().invoke(...)``
  - ``compile().get_graph()``
  - Fan-out / fan-in execution (topological order)
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore
except Exception:  # pragma: no cover - exercised in environments without langgraph
    START = "__start__"
    END = "__end__"

    @dataclass(frozen=True)
    class _GraphEdge:
        source: str
        target: str

    @dataclass(frozen=True)
    class _GraphSnapshot:
        nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]]
        edges: list[_GraphEdge]

    class _CompiledFallbackGraph:
        def __init__(
            self,
            *,
            nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
            edges: list[_GraphEdge],
            adjacency: dict[str, list[str]],
            conditionals: dict[str, tuple[Callable[[dict[str, Any]], str], dict[str, str]]],
        ) -> None:
            self._nodes = dict(nodes)
            self._edges = list(edges)
            self._adjacency = {k: list(v) for k, v in adjacency.items()}
            self._conditionals = dict(conditionals)

        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            if not isinstance(state, dict):
                raise TypeError("Graph input state must be a dict")

            current_state = dict(state)
            max_steps = max(64, len(self._nodes) * 8)
            steps = 0

            # Build layered execution plan: nodes at the same depth can
            # run in parallel (fan-out).
            layers = self._execution_layers()

            for layer in layers:
                # Filter to executable nodes only.
                runnable = [
                    name
                    for name in layer
                    if name not in (START, END)
                    and self._should_execute(name, current_state)
                ]
                if not runnable:
                    continue

                if len(runnable) == 1:
                    # Single node — run directly (no thread overhead).
                    name = runnable[0]
                    node_fn = self._nodes.get(name)
                    if node_fn is None:
                        raise KeyError(f"Graph references unknown node: {name}")
                    result = node_fn(current_state)
                    if not isinstance(result, dict):
                        raise TypeError(f"Node `{name}` must return a dict state")
                    current_state.update(result)
                    steps += 1
                else:
                    # Multiple nodes at the same depth — run in parallel.
                    snapshot = dict(current_state)
                    with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
                        futures = {}
                        for name in runnable:
                            node_fn = self._nodes.get(name)
                            if node_fn is None:
                                raise KeyError(f"Graph references unknown node: {name}")
                            futures[pool.submit(node_fn, snapshot)] = name

                        for future in as_completed(futures):
                            name = futures[future]
                            result = future.result()
                            if not isinstance(result, dict):
                                raise TypeError(
                                    f"Node `{name}` must return a dict state"
                                )
                            current_state.update(result)
                            steps += 1

                if steps > max_steps:
                    raise RuntimeError("Graph execution exceeded max step limit")

            return current_state

        def get_graph(self) -> _GraphSnapshot:
            return _GraphSnapshot(nodes=dict(self._nodes), edges=list(self._edges))

        def _execution_layers(self) -> list[list[str]]:
            """Group nodes into layers by dependency depth for parallel execution.

            Nodes in the same layer share no inter-dependencies and can run
            concurrently.
            """
            order = self._topological_order()
            # Build reverse mapping: node → set of predecessors.
            predecessors: dict[str, set[str]] = {n: set() for n in order}
            for edge in self._edges:
                predecessors.setdefault(edge.target, set()).add(edge.source)

            depth: dict[str, int] = {}
            for node in order:
                if not predecessors.get(node):
                    depth[node] = 0
                else:
                    depth[node] = max(
                        depth.get(pred, 0) for pred in predecessors[node]
                    ) + 1

            max_depth = max(depth.values()) if depth else 0
            layers: list[list[str]] = [[] for _ in range(max_depth + 1)]
            for node in order:
                layers[depth[node]].append(node)
            return layers

        def _topological_order(self) -> list[str]:
            """Compute a stable topological execution order using Kahn's algorithm."""
            # Collect all nodes including START/END.
            all_nodes: set[str] = {START, END}
            all_nodes.update(self._nodes.keys())
            for edge in self._edges:
                all_nodes.add(edge.source)
                all_nodes.add(edge.target)

            # Build in-degree map (excluding conditional edges — those are
            # resolved dynamically).
            in_degree: dict[str, int] = {node: 0 for node in all_nodes}
            forward_adj: dict[str, list[str]] = {node: [] for node in all_nodes}

            for edge in self._edges:
                # Skip edges from conditional sources — they are handled
                # dynamically in _should_execute.
                if edge.source in self._conditionals:
                    # Still register the edge for reachability.
                    forward_adj.setdefault(edge.source, [])
                    if edge.target not in forward_adj[edge.source]:
                        forward_adj[edge.source].append(edge.target)
                    in_degree.setdefault(edge.target, 0)
                    in_degree[edge.target] += 1
                else:
                    forward_adj.setdefault(edge.source, [])
                    if edge.target not in forward_adj[edge.source]:
                        forward_adj[edge.source].append(edge.target)
                    in_degree.setdefault(edge.target, 0)
                    in_degree[edge.target] += 1

            # Kahn's algorithm — start from nodes with in-degree 0.
            queue: deque[str] = deque()
            for node in sorted(all_nodes):  # sorted for deterministic order
                if in_degree.get(node, 0) == 0:
                    queue.append(node)

            order: list[str] = []
            while queue:
                node = queue.popleft()
                order.append(node)
                for neighbor in sorted(forward_adj.get(node, [])):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            return order

        def _should_execute(self, node_name: str, state: dict[str, Any]) -> bool:
            """Check if a node should execute based on conditional routing."""
            # Find all edges that target this node.
            for edge in self._edges:
                if edge.target != node_name:
                    continue

                source = edge.source
                conditional = self._conditionals.get(source)
                if conditional is None:
                    # Non-conditional edge — always execute.
                    continue

                # Conditional edge: check if the route leads to this node.
                chooser, mapping = conditional
                route = chooser(state)
                resolved_target = mapping.get(route)
                if resolved_target != node_name:
                    # The conditional chose a different branch — skip.
                    return False

            return True

    class StateGraph:
        def __init__(self, _state_type: Any) -> None:
            self._nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
            self._edges: list[_GraphEdge] = []
            self._adjacency: dict[str, list[str]] = {}
            self._conditionals: dict[
                str, tuple[Callable[[dict[str, Any]], str], dict[str, str]]
            ] = {}

        def add_node(
            self, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]
        ) -> None:
            self._nodes[str(name)] = fn

        def add_edge(self, source: str, target: str) -> None:
            src = str(source)
            dst = str(target)
            self._edges.append(_GraphEdge(source=src, target=dst))
            self._adjacency.setdefault(src, []).append(dst)

        def add_conditional_edges(
            self,
            source: str,
            condition: Callable[[dict[str, Any]], str],
            path_map: dict[str, str],
        ) -> None:
            src = str(source)
            normalized_map = {str(k): str(v) for k, v in dict(path_map).items()}
            self._conditionals[src] = (condition, normalized_map)
            for target in normalized_map.values():
                self._edges.append(_GraphEdge(source=src, target=target))

        def compile(self) -> _CompiledFallbackGraph:
            return _CompiledFallbackGraph(
                nodes=self._nodes,
                edges=self._edges,
                adjacency=self._adjacency,
                conditionals=self._conditionals,
            )
