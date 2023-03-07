from typing import Any, Dict, Union

import tarfile
import io
from pathlib import Path
from tqdm import tqdm


def load(
    path: Union[str, Path],
    verbose: Union[bool, int] = True,
) -> Dict[str, Any]:
    dictionary = {}
    with tarfile.open(path) as tar:
        for tarinfo in tqdm(tar.getmembers(), disable=not verbose):
            if tarinfo.isfile():
                key = tarinfo.name.split("/")[-1]
                value = tar.extractfile(tarinfo).read()
                dictionary[key] = value
    return dictionary


def dump(
    dictionary: Dict[str, Any],
    path: Union[str, Path],
    verbose: Union[bool, int] = True,
):
    if not isinstance(path, Path):
        path = Path(path)

    suffix = path.suffix.replace(".", "")
    dir_name = path.stem.replace(".tar", "")

    with tarfile.open(path, f"w:{suffix}") as tar:
        for key, value in tqdm(dictionary.items(), disable=not verbose):
            tarinfo = tarfile.TarInfo(f"{dir_name}/{key}")
            tarinfo.size = len(value)
            tar.addfile(tarinfo, io.BytesIO(value.encode()))
