import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from coauths.preprocessing import Preprocesser


class Transformer(Preprocesser):
    def __init__(self, data_dir: str, metadata_file: str, step_directory: str) -> None:
        super().__init__(data_dir, metadata_file, step_directory)

    def map_name_id(self, names: List[str]) -> Dict[str, int]:

        return {name: idx for idx, name in enumerate(names)}

    def set_id_column(self, s: pd.Series, map_id: Dict[str, int]) -> pd.Series:
        return s.map(lambda x: map_id.get(x, -1))

    def build_entity_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        unique_names = set(df["name"].unique())
        unique_authors = set(df["authors"].unique()).difference(unique_names)

        all_names = sorted(unique_names) + sorted(unique_authors)
        map_id = {name: i for i, name in enumerate(all_names)}

        df["nid"] = self.set_id_column(df["name"], map_id)
        df["aid"] = self.set_id_column(df["authors"], map_id)

        df = df[
            ["production_id", "name", "citation", "nid", "lattes_id", "institution",
             "production", "aid", "authors", "type", "year", "issn"]
        ]
        return df

    def build_flag_productions(self, df: pd.DataFrame) -> dict:
        flag_productions = {}
        for _, row in df.iterrows():
            name = row["nid"]
            authors = row["aid"]
            if name == authors:
                continue
            production = row["production_id"]
            if production not in flag_productions:
                flag_productions[production] = {}
            pair = frozenset([name, authors])
            if pair not in flag_productions[production]:
                flag_productions[production][pair] = False
        return flag_productions

    def build_adjacency(
        self, df: pd.DataFrame, flag_productions: dict,
        start_year: int = None, end_year: int = None,
        remove_isolated: bool = False,
    ) -> pd.DataFrame:
        if start_year is None:
            start_year = df["year"].min()
        if end_year is None:
            end_year = df["year"].max()

        nunique = df["nid"].nunique()
        nids = np.sort(df["nid"].unique())
        unique_name_set = sorted(list(df["name"].unique()))
        data_adjacency = np.zeros(shape=(nunique, nunique), dtype=np.uint32)

        if start_year == end_year:
            df_filtered = df[df["year"] == start_year].copy()
        else:
            df_filtered = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

        for _, row in df_filtered.iterrows():
            researcher_name = row["name"]
            researcher_id = row["nid"]
            citation_name = row["authors"]
            citation_id = row["aid"]
            production_id = row["production_id"]

            if citation_name in unique_name_set and citation_name != researcher_name:
                pair_key = frozenset([researcher_id, citation_id])
                if production_id in flag_productions and pair_key in flag_productions[production_id]:
                    if not flag_productions[production_id][pair_key]:
                        flag_productions[production_id][pair_key] = True
                        try:
                            data_adjacency[researcher_id][citation_id] += 1
                            data_adjacency[citation_id][researcher_id] += 1
                        except IndexError:
                            continue

        df_adjacency = pd.DataFrame(data_adjacency, columns=nids, index=nids)

        if remove_isolated:
            df_adjacency = df_adjacency[df_adjacency.sum() > 0]
            df_adjacency = df_adjacency[df_adjacency.index]

        return df_adjacency

    def frame_coauthorship_adjacency_matrix(
            self, df: pd.DataFrame, start_year: int = None, end_year: int = None,
            type: str = None, institution: str = None, remove_isolated: bool = False
            ) -> pd.DataFrame:

        if start_year is not None and end_year is not None:
            if start_year == end_year:
                df_filtered = df[df["year"] == start_year].copy()
        else:
            if start_year is None:
                start_year = df["year"].min()
            if end_year is None:
                end_year = df["year"].max()

        nunique = df["nid"].nunique()

        nids = np.sort(df["nid"].unique())

        unique_names = sorted(list(df["name"].unique()))

        data_adjacency = np.zeros(shape=(nunique, nunique), dtype=np.uint32)

        df_filtered = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

        if type is not None:
            df_filtered = df_filtered[df_filtered["type"] == type]

        if institution is not None:
            df_filtered = df_filtered[df_filtered["institution"] == institution]

        for _, row in df_filtered.iterrows():
            if row["authors"] in unique_names and row["authors"] != row["name"]:
                i = row["nid"]
                j = row["aid"]

                try:
                    data_adjacency[i][j] += 1
                    data_adjacency[j][i] += 1
                except IndexError:
                    continue

        df_adjacency = pd.DataFrame(data_adjacency, columns=nids, index=nids)

        if remove_isolated:
            df_adjacency = df_adjacency[df_adjacency.sum() > 0]
            df_adjacency = df_adjacency[df_adjacency.index]

        return df_adjacency
