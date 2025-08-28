import os
from pathlib import PurePath

def setrootdir(root: str):
    current_absolute_path: list = os.getcwd().split(str(PurePath("/")))

    if root in current_absolute_path:
        while current_absolute_path[-1] != root:
            os.chdir(PurePath("../"))
            current_absolute_path.pop()
        return f"Directory {root} successfully loaded as current working directory."
    return f"Cannot find {root} on the absolute path."
