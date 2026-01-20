import os
from pathlib import Path

from src.preprocessing import Preprocesser


class Transformer(Preprocesser):
    def __init__(self, data_dir: str, metadata_file: str, step_directory: str) -> None:
        super().__init__(data_dir, metadata_file, step_directory)
