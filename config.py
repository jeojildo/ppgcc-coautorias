import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
METADATA_FILE = DATA_DIR / "metadata.json"

PPGCC_BRAZIL_DOWNLOAD = "https://drive.google.com/uc?id=1oGxgCJAPgD0oLuIrbLLOFWLDL-wvu-oo"
PPGCC_XLSX_DOWNLOAD = "https://docs.google.com/spreadsheets/d/1gm-czw_WzNuraEGzGOMcmDn93Nh-rE7W/export?format=xlsx"

START_YEAR = 2014
END_YEAR = 2023

MIN_INSTITUTION_COAUTHORSHIPS = 50
MIN_CHORD_COAUTHORSHIPS = 150

COMMUNITY_METHOD = "leiden"
COMMUNITY_RESOLUTION = 0.6
COMMUNITY_SEED = 42
COMMUNITY_COMPARISON_METHODS = ["greedy_modularity", "louvain", "leiden"]
