import os
import sys
import xml.etree.ElementTree as et
from pathlib import Path
from typing import Dict, List, Any
import statistics
import json

import pandas as pd

from src.explattes.Pesquisador import Pesquisador

class InstitutionRegistry:
    def __init__(self, data_dir: str, selection_dir_name: str):
        self.data_dir = Path(data_dir)
        self.selection_dir_name = selection_dir_name
        self.data: Dict[str, List[Pesquisador]] = self._extract_resumes()

    def _extract_resumes(self) -> Dict[str, List]:
        selection_directory = Path(self.data_dir, self.selection_dir_name)
        resumes: Dict[str, List] = {}

        for subdir, _, files in os.walk(selection_directory):
            for file in files:
                filepath = Path(subdir, file)
                institution = filepath.parent.name

                if institution not in resumes:
                    resumes[institution] = []

                root = et.parse(filepath).getroot()
                researcher = Pesquisador(root)
                resumes[institution].append(researcher)

        return resumes

    def _stats(self):
        counts = {inst: len(lst) for inst, lst in self.data.items()}
        if not counts:
            return {
                "institutions": 0, "researchers": 0, "mean": 0, "median": 0,
                "p25": 0, "p75": 0, "min": (None, 0), "max": (None, 0), "top": []
            }

        values = list(counts.values())
        total = sum(values)
        mean = statistics.mean(values)
        median = statistics.median(values)
        p25, p75 = statistics.quantiles(values, n=4)[0], statistics.quantiles(values, n=4)[-1]
        min_inst = min(counts, key=counts.get), min(values)
        max_inst = max(counts, key=counts.get), max(values)
        top3 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]

        return {
            "institutions": len(counts),
            "researchers": total,
            "mean": mean,
            "median": median,
            "p25": p25,
            "p75": p75,
            "min": min_inst,
            "max": max_inst,
            "top": top3
        }

    def __str__(self):
        stats = self._stats()
        top_str = ", ".join([f"{inst} ({count})" for inst, count in stats["top"]])
        return (
            f"InstitutionRegistry\n"
            f"- Institutions: {stats['institutions']}\n"
            f"- Total researchers: {stats['researchers']}\n"
            f"- Researchers per institution:\n"
            f"\t- Average: {stats['mean']:.2f}\n"
            f"\t- Median: {stats['median']}\n"
            f"\t- P25: {stats['p25']}, P75: {stats['p75']}\n"
            f"- Largest: {stats['max'][0]} ({stats['max'][1]})\n"
            f"- Smallest: {stats['min'][0]} ({stats['min'][1]})\n"
            f"- Top 3 institutions: {top_str if top_str else 'None'}"
        )

    def __repr__(self):
        return self.__str__()


class Preprocesser:
    def __init__(self, data_dir: str, metadata_file: str, step_directory: str) -> None:
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.step_directory = step_directory

        os.makedirs(Path(self.data_dir, self.step_directory), exist_ok=True)

    def extract_institution_registry(self, selection_dir_name: str) -> InstitutionRegistry:
        registry = InstitutionRegistry(self.data_dir, selection_dir_name)
        return registry

    def frame_productions_institution(self, researchers: List[Pesquisador], institution: str = None) -> pd.DataFrame:
        data_productions = {
            "name": [],
            "citation": [],
            "lattes_id": [],
            "institution": [],
            "production": [],
            "authors": [],
            "location": [],
            "type": [],
            "year": [],
            "issn": [],
        }

        for researcher in researchers:
            for producao in researcher.producoes:
                data_productions["name"].append(researcher.nome)
                data_productions["citation"].append(researcher.citacoes)
                data_productions["lattes_id"].append(researcher.id)
                data_productions["institution"].append(institution)
                data_productions["production"].append(producao.titulo)
                data_productions["authors"].append(producao.autores)
                data_productions["location"].append(producao.local)
                data_productions["type"].append(producao.tipo.name)
                data_productions["year"].append(producao.ano)
                data_productions["issn"].append(producao.issn)

        df_productions = pd.DataFrame(data_productions)
        df_productions = df_productions.reset_index(names="production_id")
        return df_productions

    def frame_productions_all_institutions(self, institution_registry: InstitutionRegistry) -> pd.DataFrame:
        df_institutions = []

        for institution, researchers in institution_registry.data.items():
            df_institution = self.frame_productions_institution(researchers, institution)
            df_institutions.append(df_institution)

        df_all = pd.concat(df_institutions, ignore_index=True)
        df_all["production_id"] = df_all.index

        df_all["year"] = df_all["year"].map(lambda x: int(x) if str(x).isnumeric() else None)
        df_all = df_all.astype({"year": "Int64"})

        return df_all

    def drop_authorless_productions(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["authors"].map(len) > 0].reset_index(drop=True)
    
    def normalize_citations(authors, non_duplicated_variations, citations_reference) -> List[str]:
        normalized_authors = []
        for author in authors:
            if author in non_duplicated_variations:
                normalized_authors.append(citations_reference[author])
            else:
                normalized_authors.append(author)
        return normalized_authors
    
    def normalize_authors(self, df: pd.DataFrame) -> pd.DataFrame:
        df_citations = df.drop_duplicates(subset=["lattes_id"])[["name", "citation"]].reset_index(drop=True)
        
        citation_variations = [name for citations in df_citations["citation"] for name in citations]

        variations_counting = {}

        for citation in citation_variations:
            if citation not in variations_counting.keys():
                variations_counting[citation] = 0
            variations_counting[citation] += 1

        non_duplicated_variations = [citation for citation, count in variations_counting.items() if count == 1]
        non_duplicated_variations = set(non_duplicated_variations)

        citations_reference = {}

        for i, row in df_citations.iterrows():
            for citation in row["citation"]:
                if citation in non_duplicated_variations:
                    citations_reference[citation] = row["name"]

        df["authors"] = df["authors"].apply(lambda authors: Preprocesser.normalize_citations(authors, non_duplicated_variations, citations_reference))

        return df
    
    def explode_authors(self, df: pd.DataFrame) -> pd.DataFrame:
        df_exploded = df.explode("authors").reset_index(drop=True)
        return df_exploded
    
    def write_parquet(self, df: pd.DataFrame, step: str, name: str) -> None:
        os.makedirs(Path(self.data_dir, step), exist_ok=True)

        filepath: Path = Path(self.data_dir, step, f"{name}.parquet")
        dtype: type = type(df)

        df.to_parquet(filepath)

        new_entry: Dict[str, Any] = {
            "path": str(filepath),
            "type": str(dtype),
            "bytes": sys.getsizeof(df)
        }

        with open(self.metadata_file, 'r+') as f:
            data: Dict[str, Any] = json.load(f)

            if step not in data.keys():
                data[step] = {}

            data[step][name] = new_entry

            f.seek(0)
            json.dump(data, f, indent=4)

    def read_parquet(self, step: str, name: str) -> pd.DataFrame:

        with open(self.metadata_file, 'r') as f:
            data: Dict[str, Any] = json.load(f)

        filepath: str = data[step][name]["path"]
        df: pd.DataFrame = pd.read_parquet(filepath)
        return df
