from src.preprocessing import Preprocesser

class Visualizer(Preprocesser):
    def __init__(self, data_dir: str, metadata_file: str):
        super().__init__(data_dir, metadata_file)

