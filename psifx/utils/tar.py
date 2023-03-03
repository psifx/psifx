from typing import Any, Dict, Union

import tarfile
import io
from pathlib import Path
from tqdm import tqdm


def load(path: Union[str, Path]) -> Dict[str, Any]:
    dictionary = {}
    with tarfile.open(path) as tar:
        for tarinfo in tar.getmembers():
            if tarinfo.isfile():
                key = tarinfo.name
                value = tar.extractfile(tarinfo).read()
                dictionary[key] = value
    return dictionary


def dump(dictionary: Dict[str, Any], path: Union[str, Path]):
    if not isinstance(path, Path):
        path = Path(path)

    suffix = path.suffix.split(".")[-1]
    dir_name = path.stem.split(".")[0]

    with tarfile.open(path, f"w:{suffix}") as tar:
        for key, value in tqdm(dictionary.items()):
            tarinfo = tarfile.TarInfo(f"{dir_name}/{key}")
            tarinfo.size = len(value)
            tar.addfile(tarinfo, io.BytesIO(value.encode()))
