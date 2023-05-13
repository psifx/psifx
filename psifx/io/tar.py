from typing import Any, Dict, Union

import tarfile
import io
from pathlib import Path
from tqdm import tqdm


class TarReader(object):
    @staticmethod
    def read(
        path: Union[str, Path],
        verbose: Union[bool, int] = True,
    ) -> Dict[str, Any]:
        path = Path(path)

        assert ".tar" in path.suffixes

        with tarfile.open(path, mode="r") as tar:
            dictionary = {}
            for tarinfo in tqdm(
                tar.getmembers(),
                desc="Reading",
                disable=not verbose,
            ):
                if tarinfo.isfile():
                    key = tarinfo.name.split("/")[-1]
                    value = tar.extractfile(tarinfo).read()
                    dictionary[key] = value
        return dictionary


class TarWriter(object):
    @staticmethod
    def write(
        dictionary: Dict[str, Any],
        path: Union[str, Path],
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        path = Path(path)

        assert ".tar" in path.suffixes

        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        index = path.suffixes.index(".tar")
        if index == len(path.suffixes) - 1:
            compression = ""
        else:
            compression = path.suffixes[index + 1].replace(".", "")

        dir_name = path.stem.replace(".tar", "")
        with tarfile.open(path, mode=f"w:{compression}") as tar:
            for key, value in tqdm(
                dictionary.items(),
                desc="Writing",
                disable=not verbose,
            ):
                tarinfo = tarfile.TarInfo(name=f"{dir_name}/{key}")
                tarinfo.size = len(value)
                tar.addfile(tarinfo, io.BytesIO(value.encode()))
