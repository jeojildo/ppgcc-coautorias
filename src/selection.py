import os
import shutil
import json
from pathlib import Path
from zipfile import ZipFile


import gdown

class Selector:
    def __init__(self, data_dir: str, metadata_file: str) -> None:
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.create_metadata()

    def create_metadata(self) -> None:
        if not os.path.isfile(self.metadata_file):
            with open(self.metadata_file, 'a') as f:
                json.dump({}, f)
                f.close()

    def download(self, selection_dir_name: str, gdown_url: str) -> None:
        selection_directory = Path(self.data_dir, selection_dir_name)
        selection_file = selection_directory / "selection.zip"

        os.makedirs(selection_directory, exist_ok=True)

        gdown.download(gdown_url, str(selection_file), quiet=False)

    def extract(self, selection_dir_name: str) -> None:
        selection_directory = Path(self.data_dir, selection_dir_name)
        selection_file = selection_directory / "selection.zip"

        with ZipFile(selection_file, 'r') as zip_ref:
            zip_ref.extractall(selection_directory)
        
        os.remove(selection_file)

        for subdir, dirs, files in os.walk(selection_directory):
            for filename in (files):
                filepath = Path(subdir, filename)
                file_directory = Path(subdir, filepath.stem)
                with ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(file_directory)
                os.remove(filepath)

    def rename_xmls(self, selection_dir_name: str) -> None:
        selection_directory = Path(self.data_dir, selection_dir_name)
        
        for subdir, dirs, files in os.walk(selection_directory):
            for file in files:

                filepath = Path(subdir, file)

                if filepath.name != "curriculo.xml":
                    os.remove(filepath)
                    continue

                institution_directory = filepath.parent.parent
                lattes_directory = filepath.parent
                lattes_id = filepath.parent.name

                new_filename = f"{lattes_id}.xml"
                new_filepath = institution_directory / new_filename

                os.rename(filepath, new_filepath)
                shutil.rmtree(lattes_directory)
