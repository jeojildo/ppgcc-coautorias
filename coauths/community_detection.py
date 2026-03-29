"""Classical community detection utilities for weighted co-authorship graphs.

This module centralizes:
- execution of multiple community detection methods;
- conversion of partitions to tabular formats;
- quality metrics for each partition;
- structural and institutional profiling of communities;
- node role classification (hubs, connectors, bridges);
- institution-level network metrics;
- method-to-method comparison helpers.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
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
    mapping: dict[Any, int] = {}
    for community_index, community_nodes in enumerate(partition):
        for node in community_nodes:
            mapping[node] = community_index
    return mapping


def communities_dataframe(partition: Sequence[set[Any]]) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Edge-level community labelling
# ---------------------------------------------------------------------------

def _build_edge_dataframe(
    graph: nx.Graph,
    mapping: dict[Any, int],
    weight: str | None = "weight",
) -> pd.DataFrame:
    df_e = nx.to_pandas_edgelist(graph)
    w_col = weight if weight and weight in df_e.columns else "weight"
    df_e["c_source"] = df_e["source"].map(mapping)
    df_e["c_target"] = df_e["target"].map(mapping)
    df_e["edge_type"] = np.where(
        df_e["c_source"] == df_e["c_target"], "intra", "inter"
    )
    return df_e, w_col


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(weights: np.ndarray) -> float:
    s = weights.sum()
    if s <= 0:
        return 0.0
    p = weights / s
    return float(-(p * np.log(p + 1e-12)).sum())


def _mode_value(series: pd.Series) -> Any:
    s = series.dropna()
    if s.empty:
        return None
    return s.value_counts().index[0]


def _zscore_by_group(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sd


def _classify_role(row: pd.Series, z_hub: float = 2.5, p_connector: float = 0.62) -> str:
    z = row["z_intra"]
    p = row["participation_P"]
    if z >= z_hub and p >= p_connector:
        return "connector_hub"
    if z >= z_hub and p < p_connector:
        return "provincial_hub"
    if z < z_hub and p >= p_connector:
        return "connector"
    return "peripheral"


# ---------------------------------------------------------------------------
# Table 1 – Community structural profile  (summary)
# ---------------------------------------------------------------------------

def community_structural_profile(
    graph: nx.Graph,
    partition: Sequence[set[Any]],
    weight: str | None = "weight",
) -> pd.DataFrame:
    mapping = partition_to_mapping(partition)
    df_e, w_col = _build_edge_dataframe(graph, mapping, weight)
    df_comm = communities_dataframe(partition)

    df_intra = df_e[df_e["edge_type"] == "intra"]
    df_inter = df_e[df_e["edge_type"] == "inter"]

    intra_w = df_intra.groupby("c_source")[w_col].sum()

    ext_w = (
        df_inter.groupby("c_source")[w_col].sum()
        .add(df_inter.groupby("c_target")[w_col].sum(), fill_value=0)
    )

    partners = pd.concat([
        df_inter[["c_source", "c_target"]].rename(columns={"c_source": "c", "c_target": "p"}),
        df_inter[["c_target", "c_source"]].rename(columns={"c_target": "c", "c_source": "p"}),
    ], ignore_index=True)
    n_partners = partners.groupby("c")["p"].nunique()

    partners_w = pd.concat([
        df_inter[["c_source", "c_target", w_col]].rename(columns={"c_source": "c", "c_target": "p"}),
        df_inter[["c_target", "c_source", w_col]].rename(columns={"c_target": "c", "c_source": "p"}),
    ], ignore_index=True)
    ext_entropy = partners_w.groupby("c").apply(
        lambda g: _shannon_entropy(g[w_col].to_numpy()), include_groups=False,
    )

    summary = (
        df_comm.groupby("community").size()
        .rename("num_nodes")
        .to_frame()
        .assign(
            intra_weight=intra_w,
            external_strength_w=ext_w,
            n_partner_communities=n_partners,
            external_entropy=ext_entropy,
        )
        .fillna(0)
    )
    summary["external_share"] = (
        summary["external_strength_w"]
        / (summary["external_strength_w"] + summary["intra_weight"])
    ).fillna(0)

    return summary.sort_values("num_nodes", ascending=False).reset_index()


# ---------------------------------------------------------------------------
# Node roles – participation coefficient P, z-score, role label
# ---------------------------------------------------------------------------

def compute_node_roles(
    graph: nx.Graph,
    partition: Sequence[set[Any]],
    weight: str | None = "weight",
    z_hub: float = 2.5,
    p_connector: float = 0.62,
) -> pd.DataFrame:
    mapping = partition_to_mapping(partition)
    df_e, w_col = _build_edge_dataframe(graph, mapping, weight)
    df_intra = df_e[df_e["edge_type"] == "intra"]

    nodes_list = list(graph.nodes())
    strength_total = dict(graph.degree(weight=weight))

    intra_agg = df_intra.groupby("c_source")[w_col].sum().to_dict()
    strength_intra = {
        node: intra_agg.get(mapping.get(node, -1), 0) for node in nodes_list
    }
    strength_external = {
        node: strength_total.get(node, 0) - strength_intra.get(node, 0)
        for node in nodes_list
    }

    def _calc_participation(node: Any) -> float:
        k_total = strength_total.get(node, 0)
        if k_total == 0:
            return 0.0
        comm_weights: dict[int, float] = {}
        for _, neighbor, data in graph.edges(node, data=True):
            neigh_comm = mapping.get(neighbor)
            w = data.get(weight, 1) if weight else 1
            comm_weights[neigh_comm] = comm_weights.get(neigh_comm, 0) + w
        return 1.0 - sum((w / k_total) ** 2 for w in comm_weights.values())

    participation = {node: _calc_participation(node) for node in nodes_list}

    df = pd.DataFrame({
        "node": nodes_list,
        "community": [mapping.get(node) for node in nodes_list],
        "strength_w": [strength_total.get(node, 0) for node in nodes_list],
        "strength_intra_w": [strength_intra.get(node, 0) for node in nodes_list],
        "strength_external_w": [strength_external.get(node, 0) for node in nodes_list],
        "participation_P": [participation.get(node, 0) for node in nodes_list],
    })

    df["z_intra"] = df.groupby("community")["strength_intra_w"].transform(_zscore_by_group)
    df["role"] = df.apply(
        lambda row: _classify_role(row, z_hub=z_hub, p_connector=p_connector),
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# Table 2 – Institutional profile per community
# ---------------------------------------------------------------------------

def community_institutional_profile(
    df_nodes_roles: pd.DataFrame,
) -> pd.DataFrame:
    df = df_nodes_roles[["node", "community", "institution"]].copy()

    comm_size = df.groupby("community").size().rename("num_nodes")
    comm_inst_counts = (
        df.groupby(["community", "institution"]).size().rename("n").reset_index()
    )

    idx = comm_inst_counts.groupby("community")["n"].idxmax()
    dom = comm_inst_counts.loc[idx].set_index("community")
    dom_inst = dom["institution"].rename("dominant_institution")
    dom_share = (dom["n"] / comm_size).rename("dominant_inst_share")

    n_inst = df.groupby("community")["institution"].nunique().rename("n_institutions")

    merged = comm_inst_counts.merge(comm_size.reset_index(), on="community", how="left")
    merged["p"] = merged["n"] / merged["num_nodes"]
    merged["p_logp"] = merged["p"] * np.log(merged["p"] + 1e-12)
    inst_entropy = (-merged.groupby("community")["p_logp"].sum()).rename("institution_entropy")

    profile = pd.concat(
        [comm_size, n_inst, dom_inst, dom_share, inst_entropy], axis=1,
    ).reset_index().sort_values(
        ["n_institutions", "institution_entropy", "num_nodes"], ascending=False,
    )
    return profile


# ---------------------------------------------------------------------------
# Table 3 – Institution-level network metrics
# ---------------------------------------------------------------------------

def institution_network_metrics(
    graph: nx.Graph,
    node_to_inst: dict[Any, str],
    weight: str | None = "weight",
) -> pd.DataFrame:
    df_e = nx.to_pandas_edgelist(graph)
    w_col = weight if weight and weight in df_e.columns else "weight"

    df_e["inst_u"] = df_e["source"].map(node_to_inst)
    df_e["inst_v"] = df_e["target"].map(node_to_inst)
    df_e = df_e.dropna(subset=["inst_u", "inst_v"])

    a, b = df_e["inst_u"].to_numpy(), df_e["inst_v"].to_numpy()
    df_e["inst_min"] = np.where(a <= b, a, b)
    df_e["inst_max"] = np.where(a <= b, b, a)
    df_e["is_interinst"] = df_e["inst_min"] != df_e["inst_max"]

    inst_edges = (
        df_e.groupby(["inst_min", "inst_max"])[w_col]
        .sum()
        .reset_index(name="collab_weight")
        .sort_values("collab_weight", ascending=False)
    )

    G_inst = nx.from_pandas_edgelist(
        inst_edges, "inst_min", "inst_max", edge_attr="collab_weight",
    )

    inst_strength = dict(G_inst.degree(weight="collab_weight"))
    inst_bet = nx.betweenness_centrality(G_inst, weight="collab_weight")

    intra_w_by_inst = (
        inst_edges[inst_edges["inst_min"] == inst_edges["inst_max"]]
        .set_index("inst_min")["collab_weight"]
    )
    inst_total = pd.Series(inst_strength)
    inst_intra = intra_w_by_inst.reindex(inst_total.index).fillna(0)
    inst_external = inst_total - 2 * inst_intra
    inst_external_share = (inst_external / inst_total.replace(0, np.nan)).fillna(0)

    df_metrics = pd.DataFrame({
        "institution": list(G_inst.nodes()),
        "strength_w": pd.Series(inst_strength),
        "betweenness": pd.Series(inst_bet),
        "external_share": inst_external_share,
    }).sort_values(["strength_w", "betweenness"], ascending=False)

    return df_metrics.reset_index(drop=True)
