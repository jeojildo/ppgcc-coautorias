"""Classical community detection utilities for weighted co-authorship graphs.

This module centralizes:
- execution of multiple community detection methods;
- conversion of partitions to tabular formats;
- quality metrics for each partition;
- method-to-method comparison helpers.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, Mapping, Sequence

import networkx as nx
import pandas as pd
from networkx.algorithms.community import (
    asyn_lpa_communities,
    girvan_newman,
    greedy_modularity_communities,
    louvain_communities,
    modularity,
)
from networkx.algorithms.community.quality import partition_quality

try:
    import igraph as ig
    import leidenalg as la
except ImportError:
    ig = None
    la = None


DEFAULT_METHODS = (
    "greedy_modularity",
    "louvain",
    "leiden",
    "label_propagation",
    "girvan_newman",
)


def _normalize_partition(communities: Iterable[Iterable[Any]]) -> list[set[Any]]:
    partition = [set(community) for community in communities if len(set(community)) > 0]

    def _sort_key(nodes: set[Any]) -> tuple[int, str]:
        return (-len(nodes), min(map(str, nodes)))

    return sorted(partition, key=_sort_key)


def _edge_weight_sum(graph: nx.Graph, weight: str | None) -> float:
    if weight is None:
        return float(graph.number_of_edges())
    return float(sum(data.get(weight, 1.0) for _, _, data in graph.edges(data=True)))


def _most_valuable_edge(graph: nx.Graph, weight: str | None) -> tuple[Any, Any]:
    betweenness = nx.edge_betweenness_centrality(graph, weight=weight)
    return max(betweenness, key=betweenness.get)


def _girvan_newman_partition(
    graph: nx.Graph,
    weight: str | None,
    target_communities: int | None = None,
    max_levels: int = 20,
) -> list[set[Any]]:
    """Run Girvan-Newman and select a partition.

    If ``target_communities`` is provided, the first split reaching that number
    of communities is returned. Otherwise, the split with highest modularity up
    to ``max_levels`` is selected.
    """
    if graph.number_of_edges() == 0:
        return _normalize_partition([{node} for node in graph.nodes()])

    communities_generator = girvan_newman(
        graph,
        most_valuable_edge=lambda g: _most_valuable_edge(g, weight),
    )

    best_partition: list[set[Any]] | None = None
    best_modularity = float("-inf")

    for communities in itertools.islice(communities_generator, max_levels):
        partition = [set(community) for community in communities]

        if target_communities is not None and len(partition) >= target_communities:
            return _normalize_partition(partition)

        score = modularity(graph, partition, weight=weight)
        if score > best_modularity:
            best_modularity = score
            best_partition = partition

    if best_partition is None:
        return _normalize_partition([{node} for node in graph.nodes()])

    return _normalize_partition(best_partition)


def _leiden_partition(
    graph: nx.Graph,
    weight: str | None,
    resolution: float,
    seed: int,
) -> list[set[Any]]:
    if graph.number_of_edges() == 0:
        return _normalize_partition([{node} for node in graph.nodes()])

    if ig is None or la is None:
        raise ImportError(
            "Leiden requires optional dependencies. Install with: "
            "pip install igraph leidenalg"
        )

    node_list = list(graph.nodes())
    node_to_index = {node: index for index, node in enumerate(node_list)}

    ig_graph = ig.Graph(n=len(node_list), directed=False)
    edges_index: list[tuple[int, int]] = []
    weights: list[float] = []

    for source, target, data in graph.edges(data=True):
        edges_index.append((node_to_index[source], node_to_index[target]))
        edge_weight = 1.0 if weight is None else float(data.get(weight, 1.0))
        weights.append(edge_weight)

    if len(edges_index) > 0:
        ig_graph.add_edges(edges_index)

    partition_kwargs: Dict[str, Any] = {
        "weights": weights if weight is not None else None,
        "resolution_parameter": resolution,
    }

    try:
        partition = la.find_partition(
            ig_graph,
            la.RBConfigurationVertexPartition,
            seed=seed,
            **partition_kwargs,
        )
    except TypeError:
        # Compatibility for leidenalg versions without explicit seed support.
        partition = la.find_partition(
            ig_graph,
            la.RBConfigurationVertexPartition,
            **partition_kwargs,
        )

    return _normalize_partition([{node_list[index] for index in community} for community in partition])


def detect_communities(
    graph: nx.Graph,
    method: str = "greedy_modularity",
    weight: str | None = "n_coauthorships",
    resolution: float = 1.0,
    seed: int = 42,
    target_communities: int | None = None,
    max_levels: int = 20,
) -> list[set[Any]]:
    """Detect communities for a graph using a selected method.

    Parameters
    ----------
    graph:
        Input graph.
    method:
        One of ``greedy_modularity``, ``louvain``, ``leiden``,
        ``label_propagation``, ``girvan_newman``. Some aliases are accepted (e.g. ``greedy``,
        ``girvan``).
    weight:
        Edge attribute name used as weight. Use ``None`` for unweighted mode.
    resolution:
        Resolution parameter for ``greedy_modularity``, ``louvain`` and ``leiden``.
    seed:
        Random seed for stochastic methods (``louvain``, ``leiden``, ``label_propagation``).
    target_communities:
        Only used by ``girvan_newman``. Target number of communities.
    max_levels:
        Only used by ``girvan_newman``. Maximum decomposition levels explored.

    Returns
    -------
    list[set[Any]]
        Community partition as a list of node sets.
    """
    if graph.number_of_nodes() == 0:
        return []

    method = method.lower()
    aliases = {
        "greedy": "greedy_modularity",
        "cnm": "greedy_modularity",
        "leidenalg": "leiden",
        "asyn_lpa": "label_propagation",
        "labelpropagation": "label_propagation",
        "girvan": "girvan_newman",
    }
    method = aliases.get(method, method)

    if method == "greedy_modularity":
        communities = greedy_modularity_communities(
            graph,
            weight=weight,
            resolution=resolution,
        )
        return _normalize_partition(communities)

    if method == "louvain":
        communities = louvain_communities(
            graph,
            weight=weight,
            resolution=resolution,
            seed=seed,
        )
        return _normalize_partition(communities)

    if method == "leiden":
        return _leiden_partition(
            graph=graph,
            weight=weight,
            resolution=resolution,
            seed=seed,
        )

    if method == "label_propagation":
        communities = asyn_lpa_communities(
            graph,
            weight=weight,
            seed=seed,
        )
        return _normalize_partition(communities)

    if method == "girvan_newman":
        return _girvan_newman_partition(
            graph=graph,
            weight=weight,
            target_communities=target_communities,
            max_levels=max_levels,
        )

    available = ", ".join(DEFAULT_METHODS)
    raise ValueError(f"Unknown community method: '{method}'. Available methods: {available}.")


def partition_to_mapping(partition: Sequence[set[Any]]) -> dict[Any, int]:
    """Convert a partition into node -> community_id mapping."""
    mapping: dict[Any, int] = {}
    for community_index, community_nodes in enumerate(partition):
        for node in community_nodes:
            mapping[node] = community_index
    return mapping


def communities_dataframe(partition: Sequence[set[Any]]) -> pd.DataFrame:
    """Return a DataFrame with columns: ``node`` and ``community``."""
    rows: list[dict[str, Any]] = []
    for community_index, community_nodes in enumerate(partition):
        for node in sorted(community_nodes, key=str):
            rows.append({"node": node, "community": community_index})
    return pd.DataFrame(rows)


def community_report(
    graph: nx.Graph,
    partition: Sequence[set[Any]],
    weight: str | None = "n_coauthorships",
) -> dict[str, float]:
    """Compute quality and structure metrics for a partition.

    Returned fields include:
    ``modularity``, ``coverage``, ``performance``, community size stats,
    and intra/inter-community edge-weight ratios.
    """
    if graph.number_of_nodes() == 0:
        return {
            "num_nodes": 0.0,
            "num_edges": 0.0,
            "num_communities": 0.0,
            "largest_community": 0.0,
            "smallest_community": 0.0,
            "mean_community_size": 0.0,
            "modularity": 0.0,
            "coverage": 0.0,
            "performance": 0.0,
            "intra_weight_ratio": 0.0,
            "inter_weight_ratio": 0.0,
        }

    mapping = partition_to_mapping(partition)
    total_weight = _edge_weight_sum(graph, weight)
    intra_weight = 0.0
    inter_weight = 0.0

    for source, target, data in graph.edges(data=True):
        edge_weight = data.get(weight, 1.0) if weight is not None else 1.0
        if mapping[source] == mapping[target]:
            intra_weight += edge_weight
        else:
            inter_weight += edge_weight

    if graph.number_of_edges() > 0 and len(partition) > 1:
        modularity_score = float(modularity(graph, partition, weight=weight))
    else:
        modularity_score = 0.0

    try:
        coverage, performance = partition_quality(graph, partition)
    except Exception:
        coverage, performance = 0.0, 0.0

    sizes = [len(community_nodes) for community_nodes in partition] or [0]

    return {
        "num_nodes": float(graph.number_of_nodes()),
        "num_edges": float(graph.number_of_edges()),
        "num_communities": float(len(partition)),
        "largest_community": float(max(sizes)),
        "smallest_community": float(min(sizes)),
        "mean_community_size": float(sum(sizes) / len(sizes)),
        "modularity": modularity_score,
        "coverage": float(coverage),
        "performance": float(performance),
        "intra_weight_ratio": float(intra_weight / total_weight) if total_weight > 0 else 0.0,
        "inter_weight_ratio": float(inter_weight / total_weight) if total_weight > 0 else 0.0,
    }


def compare_community_algorithms(
    graph: nx.Graph,
    methods: Sequence[str] = DEFAULT_METHODS,
    method_params: Mapping[str, Mapping[str, Any]] | None = None,
    default_weight: str | None = "n_coauthorships",
) -> pd.DataFrame:
    """Run multiple methods and return a comparison table.

    Parameters
    ----------
    graph:
        Input graph.
    methods:
        Sequence of methods to evaluate.
    method_params:
        Optional method-specific kwargs. Example::

            {
                "louvain": {"resolution": 1.0, "seed": 42},
                "leiden": {"resolution": 1.0, "seed": 42},
                "girvan_newman": {"target_communities": 4, "max_levels": 25},
            }

    default_weight:
        Default edge weight attribute name. Can be overridden per method by
        passing ``weight`` inside ``method_params[method]``.
    """
    rows: list[dict[str, Any]] = []
    method_params = method_params or {}

    for method in methods:
        params = dict(method_params.get(method, {}))
        weight = params.pop("weight", default_weight)

        partition = detect_communities(
            graph,
            method=method,
            weight=weight,
            **params,
        )

        metrics = community_report(graph, partition, weight=weight)
        metrics["method"] = method
        rows.append(metrics)

    columns = [
        "method",
        "num_nodes",
        "num_edges",
        "num_communities",
        "largest_community",
        "smallest_community",
        "mean_community_size",
        "modularity",
        "coverage",
        "performance",
        "intra_weight_ratio",
        "inter_weight_ratio",
    ]

    comparison = pd.DataFrame(rows, columns=columns)
    return comparison.sort_values(by="modularity", ascending=False).reset_index(drop=True)
