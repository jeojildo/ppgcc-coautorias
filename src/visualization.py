from pathlib import Path
from typing import Tuple, Callable, Dict
import string
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.drawing.layout import rescale_layout

import holoviews as hv
from bokeh.io.export import export_png, export_svgs

from src.preprocessing import Preprocesser
from src.utils import get_colors, combine_rgb, hex2rgb

class Visualizer(Preprocesser):
    def __init__(self, data_dir: str, metadata_file: str, step_directory: str) -> None:
        super().__init__(data_dir, metadata_file, step_directory)
        self.figure_directory = Path(self.data_dir, self.step_directory)

    def frame_yearly_publications(self, df: pd.DataFrame, start_year: int = None, end_year: int = None, institution: str = None) -> pd.DataFrame:
        if start_year is None:
            start_year = df["year"].min()
        if end_year is None:
            end_year = df["year"].max()

        df_yearly_publications = df.copy()
        df_yearly_publications = df_yearly_publications[(df_yearly_publications["year"] >= start_year) & (df_yearly_publications["year"] <= end_year)]
        df_yearly_publications = df_yearly_publications[df_yearly_publications["institution"] == institution] if institution else df_yearly_publications
        df_yearly_publications = df_yearly_publications[["year", "type"]]
        df_yearly_publications = df_yearly_publications[df_yearly_publications["type"].isin(["CONFERENCIA", "PERIODICO"])]
        df_yearly_publications = df_yearly_publications.groupby(["year", "type"]).size().reset_index(name="count")
        df_yearly_publications = df_yearly_publications.pivot(index="year", columns="type", values="count")
        
        return df_yearly_publications
    
    def plot_yearly_publications(self, df_yearly: pd.DataFrame, figsize: tuple = (8, 5), filename: str = "yearly_publications") -> Tuple[plt.Figure, plt.Axes]:
        fig_yearly_publications, ax_yearly_publications = plt.subplots(figsize=figsize)
        df_yearly.plot(kind="line", ax=ax_yearly_publications, color=["lightcoral", "skyblue"], marker='o', linewidth=2, markersize=8)

        ax_yearly_publications.set_xlabel("Ano", fontsize=12)
        ax_yearly_publications.set_ylabel("Total de publicações", fontsize=12)
        ax_yearly_publications.legend(title="Tipo de publicação", labels=["Conferência", "Periódico"], fontsize=12, title_fontsize=12)

        for line in ax_yearly_publications.get_lines():
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                ax_yearly_publications.text(x, y+50, f"{y}", fontsize=12, ha='center', va='bottom')

        ax_yearly_publications.yaxis.grid(linestyle='--', which='major', color='grey', alpha=.25)
        ax_yearly_publications.xaxis.grid(linestyle='--', which='major', color='grey', alpha=.25)

        ax_yearly_publications.set_xticks(df_yearly.index)
        ax_yearly_publications.set_xticklabels(df_yearly.index, rotation=0, fontsize=12)
        ax_yearly_publications.set_yticklabels(ax_yearly_publications.get_yticks().astype(int), fontsize=12)

        plt.tight_layout()

        plt.savefig(Path(self.figure_directory, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.svg"), format='svg', bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.pdf"), format='pdf', bbox_inches='tight')

        return fig_yearly_publications, ax_yearly_publications
    
    def frame_yearly_coauthorships(self, df: pd.DataFrame, start_year: int = None, end_year: int = None, institution: str = None) -> pd.DataFrame:
        if start_year is None:
            start_year = df["year"].min()
        if end_year is None:
            end_year = df["year"].max()

        df_yearly_coauthorships = df.copy()
        df_yearly_coauthorships = df_yearly_coauthorships[(df_yearly_coauthorships["year"] >= start_year) & (df_yearly_coauthorships["year"] <= end_year)]
        df_yearly_coauthorships = df_yearly_coauthorships[df_yearly_coauthorships["institution"] == institution] if institution else df_yearly_coauthorships
        df_yearly_coauthorships = df_yearly_coauthorships[["name", "year", "authors", "type"]]
        df_yearly_coauthorships["coauthors"] = df_yearly_coauthorships.apply(lambda row: [author for author in row["authors"] if author != row["name"]], axis=1)
        df_yearly_coauthorships["n_coauthors"] = df_yearly_coauthorships["coauthors"].apply(len)
        df_yearly_coauthorships = df_yearly_coauthorships[df_yearly_coauthorships["type"].isin(["CONFERENCIA", "PERIODICO"])]
        df_yearly_coauthorships = df_yearly_coauthorships[["year", "type", "n_coauthors"]]
        df_yearly_coauthorships = df_yearly_coauthorships.groupby(["year", "type"])["n_coauthors"].sum().reset_index()
        df_yearly_coauthorships = df_yearly_coauthorships.pivot(index="year", columns="type", values="n_coauthors")
        
        return df_yearly_coauthorships

    def plot_yearly_coauthorships(self, df: pd.DataFrame, figsize: tuple = (8, 5), filename: str = "yearly_coauthorships") -> Tuple[plt.Figure, plt.Axes]:
        fig_yearly_coauthorships, ax_yearly_coauthorships = plt.subplots(figsize=figsize)

        df.plot(
            kind="line", 
            ax=ax_yearly_coauthorships, 
            color=["lightcoral", "skyblue"], marker='o', linewidth=2, markersize=8
        )

        ax_yearly_coauthorships.set_xlabel("Ano", fontsize=12)
        ax_yearly_coauthorships.set_ylabel("Total de coautores", fontsize=12)
        ax_yearly_coauthorships.legend(title="Tipo de publicação", labels=["Conferência", "Periódico"], fontsize=12, title_fontsize=12)

        for line in ax_yearly_coauthorships.get_lines():
            label = line.get_label()
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                if label == "CONFERENCIA":
                    ax_yearly_coauthorships.text(x, y+500, f"{y}", fontsize=12, ha='center', va='bottom')
                elif label == "PERIODICO":
                    ax_yearly_coauthorships.text(x, y-1200, f"{y}", fontsize=12, ha='center', va='bottom')

        ax_yearly_coauthorships.yaxis.grid(linestyle='--', which='major', color='grey', alpha=.25)
        ax_yearly_coauthorships.xaxis.grid(linestyle='--', which='major', color='grey', alpha=.25)

        ax_yearly_coauthorships.set_xticklabels(ax_yearly_coauthorships.get_xticklabels(), rotation=0, fontsize=12)
        ax_yearly_coauthorships.set_yticklabels(ax_yearly_coauthorships.get_yticks().astype(int), fontsize=12)

        plt.tight_layout()
        plt.savefig(Path(self.figure_directory, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.svg"), format='svg', bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.pdf"), format='pdf', bbox_inches='tight')
        
        return fig_yearly_coauthorships, ax_yearly_coauthorships
    
    def frame_coauthorships_institution(self, df: pd.DataFrame, institution: str) -> pd.DataFrame:
        df_coauthorships_institution = df.copy()
        df_coauthorships_institution = df_coauthorships_institution[df_coauthorships_institution["institution"] == institution]
        df_coauthorships_institution = df_coauthorships_institution[["name", "authors"]]
        df_coauthorships_institution["coauthors"] = df_coauthorships_institution.apply(lambda row: [author for author in row["authors"] if author != row["name"]], axis=1)
        df_coauthorships_institution = df_coauthorships_institution[["name", "coauthors"]]
        df_coauthorships_institution = df_coauthorships_institution.explode("coauthors")
        df_coauthorships_institution = df_coauthorships_institution.drop_duplicates(subset=["name", "coauthors"])
        df_coauthorships_institution = df_coauthorships_institution.groupby(by="name").size().reset_index(name="n_coauthorships")
        df_coauthorships_institution = df_coauthorships_institution.sort_values(by="n_coauthorships", ascending=False)

        return df_coauthorships_institution
    
    def plot_coauthorships_institution(self, df: pd.DataFrame, figsize: tuple = (8, 5), filename: str = "coauthorships_institution") -> Tuple[plt.Figure, plt.Axes]:
        fig_coauthorships_institution, ax_coauthorships_institution = plt.subplots(figsize=figsize)

        sns.barplot(data=df, x="n_coauthorships", y="name", ax=ax_coauthorships_institution, palette="viridis")

        for index, value in enumerate(df["n_coauthorships"]):
            ax_coauthorships_institution.text(value + 1, index, str(value), color='black', va='center')

        ax_coauthorships_institution.set_xlabel("Número de coautorias")
        ax_coauthorships_institution.set_ylabel("Professor")

        plt.tight_layout()

        plt.savefig(Path(self.figure_directory, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.svg"), format='svg', bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.pdf"), format='pdf', bbox_inches='tight')

        return fig_coauthorships_institution, ax_coauthorships_institution

    def publications_by_researcher(self, df: pd.DataFrame, institution: str) -> pd.DataFrame:
        df_publications_by_researcher = df[["name", "type"]]
        df_publications_by_researcher = df_publications_by_researcher[df["institution"] == institution]
        df_publications_by_researcher = df_publications_by_researcher[df_publications_by_researcher["type"].isin(["CONFERENCIA", "PERIODICO"])]
        df_publications_by_researcher = df_publications_by_researcher.groupby(["name", "type"]).size().reset_index(name="count")
        df_publications_by_researcher = df_publications_by_researcher.pivot(index="name", columns="type", values="count").fillna(0)
        df_publications_by_researcher["total"] = df_publications_by_researcher.sum(axis=1)
        df_publications_by_researcher = df_publications_by_researcher.sort_values(by=["total", "PERIODICO", "CONFERENCIA"], ascending=False).reset_index()
        
        return df_publications_by_researcher

    def plot_publications_by_researcher(self, df: pd.DataFrame, figsize: tuple = (10, 6), filename: str = "publications_by_researcher") -> Tuple[plt.Figure, plt.Axes]:
        df = df.sort_values("total", ascending=True)

        y = range(len(df))
        conf = df["CONFERENCIA"].to_numpy()
        peri = df["PERIODICO"].to_numpy()
        labels = df["name"].tolist()
        totals = (conf + peri)

        fig_publications_by_researcher, ax_publications_by_researcher = plt.subplots(figsize=figsize)

        bars_conferences = ax_publications_by_researcher.barh(y, conf, color="lightcoral", label="Conferência")
        bars_periodicals = ax_publications_by_researcher.barh(y, peri, left=conf, color="skyblue", label="Periódico")

        ax_publications_by_researcher.set_yticks(y)
        ax_publications_by_researcher.set_yticklabels(labels)

        ax_publications_by_researcher.set_xlabel("Número de publicações")
        ax_publications_by_researcher.set_ylabel("Professor")
        ax_publications_by_researcher.legend(title="Tipo de publicação")


        ax_publications_by_researcher.xaxis.grid(linestyle='--', which='major', color='grey', alpha=.25)
        ax_publications_by_researcher.yaxis.grid(False)


        for i, (c, p, t) in enumerate(zip(conf, peri, totals)):
            if c > 0:
                ax_publications_by_researcher.text(c / 2, i, f"{int(c)}", fontsize=7, ha='center', va='center')
            if p > 0:
                ax_publications_by_researcher.text(c + p / 2, i, f"{int(p)}", fontsize=7, ha='center', va='center')
            ax_publications_by_researcher.text(t + max(totals) * 0.01 + 1, i, f"{int(t)}", fontsize=7, ha='left', va='center')

        plt.tight_layout()

        plt.savefig(Path(self.figure_directory, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.svg"), format='svg', bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.pdf"), format='pdf', bbox_inches='tight')

        return fig_publications_by_researcher, ax_publications_by_researcher
    
    def frame_coauthorship_network(self, df: pd.DataFrame, researcher_aliases: Dict[str, str]) -> pd.DataFrame:
        df_coauthorship_network = df[df["name"].isin(set(researcher_aliases.keys()))]
        df_coauthorship_network = df_coauthorship_network[["name", "authors"]]
        df_coauthorship_network = df_coauthorship_network.explode("authors")
        df_coauthorship_network = df_coauthorship_network[df_coauthorship_network["name"] != df_coauthorship_network["authors"]]
        df_coauthorship_network = df_coauthorship_network[df_coauthorship_network["authors"].isin(set(researcher_aliases.keys()))]
        df_coauthorship_network = df_coauthorship_network.map(lambda x: researcher_aliases[x] if x in researcher_aliases else x)
        df_coauthorship_network = df_coauthorship_network.groupby(by=["name", "authors"]).size().reset_index(name="n_coauthorships")
        df_coauthorship_network.columns = ["source", "target", "n_coauthorships"]
        df_coauthorship_network = df_coauthorship_network[df_coauthorship_network["n_coauthorships"] > 1]

        return df_coauthorship_network
    
    def plot_coauthorship_network(self, df: pd.DataFrame, figsize: tuple = (10, 10), filename: str = "coauthorship_network") -> plt.Figure:
        G = nx.from_pandas_edgelist(
            df,
            source="source",
            target="target",
            edge_attr="n_coauthorships"
        )

        weights = [d["n_coauthorships"] for _, _, d in G.edges(data=True)]
        edge_widths = [np.log1p(w) for w in weights]

        communities = list(greedy_modularity_communities(G))
        communities = sorted(communities, key=lambda c: (-len(c), sorted(c)[0]))

        node_to_comm = {}
        for idx, comm in enumerate(communities):
            for n in comm:
                node_to_comm[n] = idx

        n_comms = len(communities)
        palette = sns.color_palette("pastel", n_comms)
        node_colors = [palette[node_to_comm[n]] for n in G.nodes]

        CG = nx.Graph()
        CG.add_nodes_from(range(n_comms))
        for u, v, d in G.edges(data=True):
            cu = node_to_comm[u]
            cv = node_to_comm[v]
            if cu != cv:
                w = d.get("n_coauthorships", 1)
                if CG.has_edge(cu, cv):
                    CG[cu][cv]["weight"] += w
                else:
                    CG.add_edge(cu, cv, weight=w)

        pos_comm = nx.kamada_kawai_layout(CG, weight="weight") if CG.number_of_edges() > 0 else {
            i: (np.cos(2*np.pi*i/n_comms), np.sin(2*np.pi*i/n_comms)) for i in range(n_comms)
        }

        pos = {}
        max_size = max(len(c) for c in communities) if communities else 1
        for ci, comm in enumerate(communities):
            subG = G.subgraph(comm).copy()
            sub_pos = nx.spring_layout(subG, k=1)
            sub_pos = {n: np.array(p) for n, p in sub_pos.items()}
            arr = np.array(list(sub_pos.values()))
            arr = rescale_layout(arr, scale=1.0)
            scale = 0.25 + 0.15*(len(subG)/max_size)
            center = np.array(pos_comm[ci])
            for (n, _), p in zip(sub_pos.items(), arr):
                pos[n] = center + p*scale

        fig_coauthorship_network = plt.figure(figsize=figsize)

        nx.draw_networkx_nodes(
            G, pos,
            node_size=200,
            node_color=node_colors
        )

        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color="gray",
            alpha=0.6
        )

        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
        )

        edge_labels = nx.get_edge_attributes(G, "n_coauthorships")
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="white", alpha=0.3)
        )

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(Path(self.figure_directory, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.svg"), format='svg', bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.pdf"), format='pdf', bbox_inches='tight')
        
        return fig_coauthorship_network
    
    def frame_institution_network(self, df: pd.DataFrame, start_year: int, end_year: int, min_coauthorships: int) -> pd.DataFrame:
        df_authors_institution = df.copy()
        df_authors_institution = df_authors_institution[["name", "production_id", "institution", "authors", "year"]]
        df_authors_institution = df_authors_institution[(df_authors_institution["year"] >= start_year) & (df_authors_institution["year"] <= end_year)]
        df_authors_institution = df_authors_institution.drop(columns=["year"])
        df_authors_institution = df_authors_institution.groupby(["production_id", "name", "institution"]).agg(list).reset_index()

        author_to_institution = {row["name"]: row["institution"] for i, row in df_authors_institution[["name", "institution"]].drop_duplicates().iterrows()}

        df_authors_institution["authors_institution"] = df_authors_institution["authors"].apply(
            lambda authors: [author_to_institution[author] for author in authors if author in author_to_institution]
        )

        df_authors_institution = df_authors_institution[df_authors_institution["authors_institution"].apply(len) > 1]
        df_authors_institution["has_all_authors_institution"] = df_authors_institution.apply(lambda row: len(row["authors_institution"]) == len(row["authors"]), axis=1)
        
        df_authors_institution = df_authors_institution[df_authors_institution["has_all_authors_institution"]]
        df_authors_institution = df_authors_institution[["production_id", "authors_institution"]]

        def powerset(iterable, degree):
            return list(combinations(iterable, degree))
        
        institution_relations = {}

        for index, row in df_authors_institution.iterrows():
            institutions = row["authors_institution"]
            if len(institutions) < 2:
                continue
            pairs = powerset(institutions, 2)
            for pair in pairs:
                pair = tuple(sorted(pair))
                if pair in institution_relations:
                    institution_relations[pair] += 1
                else:
                    institution_relations[pair] = 1

        data_institution_relations = {
            "source": [],
            "target": [],
            "n_coauthorships": []
        }

        for (inst1, inst2), n_coauth in institution_relations.items():
            data_institution_relations["source"].append(inst1)
            data_institution_relations["target"].append(inst2)
            data_institution_relations["n_coauthorships"].append(n_coauth)

        df_institution_relations = pd.DataFrame(data_institution_relations)
        df_institution_relations = df_institution_relations.sort_values(by="n_coauthorships", ascending=False).reset_index(drop=True)

        df_institution_relations = df_institution_relations[df_institution_relations["source"] != df_institution_relations["target"]]
        df_institution_relations = df_institution_relations[df_institution_relations["n_coauthorships"] >= min_coauthorships]

        return df_institution_relations

    def plot_institution_network(self, df: pd.DataFrame, figsize: tuple = (10, 10), filename: str = "institution_network") -> plt.Figure:
        G = nx.from_pandas_edgelist(
            df,
            source="source",
            target="target",
            edge_attr="n_coauthorships"
        )

        communities = list(greedy_modularity_communities(G))
        communities = sorted(communities, key=lambda c: (-len(c), sorted(c)[0]))

        node_to_comm = {}
        for idx, comm in enumerate(communities):
            for n in comm:
                node_to_comm[n] = idx

        n_comms = len(communities)
        palette = sns.color_palette("pastel", n_comms)

        C = nx.Graph()
        C.add_nodes_from(range(n_comms))
        for u, v, d in G.edges(data=True):
            cu, cv = node_to_comm[u], node_to_comm[v]
            if cu != cv:
                w = d.get("n_coauthorships", 1)
                if C.has_edge(cu, cv):
                    C[cu][cv]["weight"] += w
                else:
                    C.add_edge(cu, cv, weight=w)

        pos_comm = nx.circular_layout(C, scale=5.0)

        pos = {}
        for c_idx, comm in enumerate(communities):
            sub = G.subgraph(comm)

            base = 1.0 / np.sqrt(max(len(sub), 1))
            k_sub = max(1.2, base * 2.0)

            pos_sub = nx.spring_layout(
                sub,
                k=k_sub,
                iterations=200,
                # seed=42,
                weight="n_coauthorships"
            )

            sub_xy = np.array(list(pos_sub.values()))
            if len(sub_xy) > 0:
                sub_xy = sub_xy - sub_xy.mean(axis=0, keepdims=True)
                scale = 1.0 + 0.35 * np.log1p(len(sub))
                sub_xy = sub_xy * scale

            cx, cy = pos_comm.get(c_idx, (0.0, 0.0))
            for i, n in enumerate(sub.nodes()):
                if len(sub_xy) > 0:
                    pos[n] = (sub_xy[i, 0] + cx, sub_xy[i, 1] + cy)
                else:
                    pos[n] = (cx, cy)


        weights = [d["n_coauthorships"] for _, _, d in G.edges(data=True)]
        edge_widths = [np.log1p(w) for w in weights]

        fig_institution_network = plt.figure(figsize=figsize)

        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color="gray",
            alpha=0.6,
        )

        node_colors = {n: palette[node_to_comm[n]] for n in G.nodes}
        labels = {n: str(n) for n in G.nodes}
        for n, (x, y) in pos.items():
            plt.text(
                x, y, labels[n],
                ha="center", va="center",
                fontsize=10,
                bbox=dict(boxstyle="square,pad=0.28", fc=node_colors[n], ec="black", lw=0.6)
            )

        edge_labels = nx.get_edge_attributes(G, "n_coauthorships")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, rotate=False, bbox=dict(fc="white", ec="white", lw=0.5, alpha=0.3))

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(Path(self.figure_directory, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.svg"), format='svg', bbox_inches='tight')
        plt.savefig(Path(self.figure_directory, f"{filename}.pdf"), format='pdf', bbox_inches='tight')
        
        return fig_institution_network
    
    def plot_sankey_researcher(self, df: pd.DataFrame) -> None:

        labels = df["source"].tolist() + df["target"].tolist()
        labels = sorted(list(set(labels)))

        labels_directional = [f"S-{label}" for label in labels] + [f"T-{label}" for label in labels]
        labels_clean = [label.split("-")[1] for label in labels_directional]

        colors = get_colors(len(labels))
        color_mapping = {l: c for l, c in zip(labels_directional, colors+colors)}

        edge_colors = []

        for i, row in df.iterrows():
            color_1 = color_mapping[f"S-{row['source']}"]
            color_2 = color_mapping[f"T-{row['target']}"]

            color_1 = hex2rgb(color_1)
            color_2 = hex2rgb(color_2)

            edge_color = combine_rgb([color_1, color_2])
            edge_colors.append(f"rgb{edge_color}")

        for s_prof in sorted(df["source"].unique()):
            df_unique = df[df["source"] == s_prof]

            source_indices = [labels_directional.index(f"S-{src}") for src in df_unique["source"]]
            source_colors = [color_mapping[f"S-{src}"] for src in df_unique["source"]]

            target_indices = [labels_directional.index(f"T-{tgt}") for tgt in df_unique["target"]]
            target_colors = [color_mapping[f"T-{tgt}"] for tgt in df_unique["target"]]

            values = df_unique["n_coauthorships"].tolist()

            fig_sankey = go.Figure(
                data=[
                    go.Sankey(
                        node = dict(
                            pad = 15,
                            thickness = 20,
                            line = dict(color="black", width=0.5),
                            label = labels_clean,
                            color = list(color_mapping.values())
                        ),
                        link = dict(
                            source = source_indices,
                            target = target_indices,
                            value = values,
                            color = edge_colors
                        )
                    )
                ]
            )

            fig_sankey.update_layout(
                width=1280,
                height=720,
                title_text=f"Coautorias do professor {s_prof}",
                title_x=0.5
            )

            fig_sankey.write_image(Path(self.figure_directory, "sankey", f"sankey_{s_prof}.png"), scale=2)

    def frame_institution_coauthorship(self, df: pd.DataFrame, start_year: int = 2014, end_year: int = 2023, n_coauthorships: int = 200) -> pd.DataFrame:
        institution_reference = {row["name"]: row["institution"] for _, row in df.iterrows()}

        df_institution_coauthorship = df[["production_id", "authors", "year"]]
        df_institution_coauthorship = df_institution_coauthorship[(df_institution_coauthorship["year"] >= start_year) & (df_institution_coauthorship["year"] <= end_year)]
        df_institution_coauthorship = df_institution_coauthorship.drop(columns=["year"])
        df_institution_coauthorship["institutions"] = df_institution_coauthorship["authors"].apply(lambda author: institution_reference[author] if author in institution_reference else False)
        df_institution_coauthorship = df_institution_coauthorship[df_institution_coauthorship["institutions"] != False]
        df_institution_coauthorship = df_institution_coauthorship.groupby(["production_id"]).agg(list).reset_index()
        df_institution_coauthorship = df_institution_coauthorship.drop(columns=["authors"])
        df_institution_coauthorship["institutions"] = df_institution_coauthorship["institutions"].apply(set).apply(list)
        df_institution_coauthorship = df_institution_coauthorship[df_institution_coauthorship["institutions"].apply(len) > 1]

        df_institution_coauthorship["institutions_string"] = df_institution_coauthorship["institutions"].map(lambda x: " - ".join(x))
        df_institution_coauth_count = df_institution_coauthorship.groupby("institutions_string").size().sort_values(ascending=False).reset_index(name="count")
        df_institution_coauth_count = df_institution_coauth_count[df_institution_coauth_count["count"] >= n_coauthorships]
        
        df_institution_coauthorship = df_institution_coauthorship[df_institution_coauthorship["institutions_string"].isin(df_institution_coauth_count["institutions_string"])]

        return df_institution_coauthorship
    
    def plot_chord_institution(self, df: pd.DataFrame, width: int = 600, height: int = 600, filename: str = "institution_coauthorship_network") -> None:
        hv.extension("bokeh")

        edge_counts = Counter()
        all_nodes = set()

        for institutions in df["institutions"]:
            unique_insts = {str(x).strip() for x in institutions if pd.notna(x) and str(x).strip()}
            all_nodes.update(unique_insts)
            for u, v in combinations(sorted(unique_insts), 2):
                edge_counts[(u, v)] += 1

        G = nx.Graph()
        G.add_nodes_from(sorted(all_nodes))
        for (u, v), w in edge_counts.items():
            G.add_edge(u, v, weight=w)

        MIN_W = 1
        to_drop = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < MIN_W]
        G.remove_edges_from(to_drop)
        G.remove_nodes_from(list(nx.isolates(G)))

        if len(G) == 0 or G.number_of_edges() == 0:
            chord = hv.Chord([]).opts(width=width, height=height)
        else:
            nodes_sorted = sorted(G.nodes())
            idx = {n: i for i, n in enumerate(nodes_sorted)}

            nodes_df = pd.DataFrame({
                "index": [idx[n] for n in nodes_sorted],
                "name": nodes_sorted,
            })

            links_df = pd.DataFrame([
                {"source": idx[u], "target": idx[v], "value": d.get("weight", 1)}
                for u, v, d in G.edges(data=True)
            ])

            nodes_ds = hv.Dataset(nodes_df, kdims="index")

            chord = hv.Chord((links_df, nodes_ds)).opts(
                width=width,
                height=height,
                labels="name",
                cmap="Category20",
                edge_color="source",
                node_color="index",
            )

        renderer = hv.renderer("bokeh")
        bokeh_plot = renderer.get_plot(chord).state

        png_path = Path(self.figure_directory, f"{filename}.png")
        export_png(bokeh_plot, filename=str(png_path))

        bokeh_plot.output_backend = "svg"
        svg_path = Path(self.figure_directory, f"{filename}.svg")
        export_svgs(bokeh_plot, filename=str(svg_path))

        try:
            import cairosvg
            pdf_path = Path(self.figure_directory, f"{filename}.pdf")
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        except Exception:
            pass

        return chord


def generate_labels(n: int) -> list[str]:
    if not 1 <= n <= 50:
        raise ValueError("n must be between 1 and 50 (26 Latin + 24 Greek).")

    latin = list(string.ascii_uppercase)
    greek = [chr(c) for c in range(0x03B1, 0x03C9 + 1)]
    all_labels = latin + greek

    return all_labels[:n]
