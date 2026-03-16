import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from src.preprocessing import Preprocesser


class Transformer(Preprocesser):
    def __init__(self, data_dir: str, metadata_file: str, step_directory: str) -> None:
        super().__init__(data_dir, metadata_file, step_directory)

    def map_name_id(self, names: List[str]) -> Dict[str, int]:

        return {name: idx for idx, name in enumerate(names)}

    def set_id_column(self, s: pd.Series, map_id: Dict[str, int]) -> pd.Series:
        return s.map(lambda x: map_id.get(x, -1))

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
    