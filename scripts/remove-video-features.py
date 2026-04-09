from typing import Sequence
import json, pathlib
import itertools
from sys import argv

def main(paths: Sequence[str]):
    all_info_file_paths = itertools.chain.from_iterable(
        pathlib.Path(p).rglob("info.json")
        for p in paths
    )
    for info_file_path in all_info_file_paths:
        with open(info_file_path, "r") as f:
            info = json.load(f)
        info["features"] = {k: v for k, v in info["features"].items() if v.get("dtype", "") != "video"}
        with open(info_file_path, "w") as f:
            json.dump(info, f, indent=4)

if __name__ == "__main__":
    main(argv[1:])