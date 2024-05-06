"""TAR I/O module."""

from typing import Any, Dict, Union

import tarfile
import io
from pathlib import Path
from tqdm import tqdm


class TarReader:
    """
    Safe TAR reader
    """

    @staticmethod
    def check(path: Union[str, Path]):
        """
        Checks that a file has of the correct extension and exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if ".tar" not in path.suffixes:
            raise NameError(path)
        if not path.exists():
            raise FileNotFoundError(path)

    @staticmethod
    def read(
        path: Union[str, Path],
        verbose: Union[bool, int] = True,
    ) -> Dict[str, Any]:
        """
        Extracts the content from a TAR archive.

        :param path: Path to the file.
        :param verbose: Verbosity of the method.
        :return: Extracted data.
        """
        path = Path(path)
        TarReader.check(path=path)

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


class TarWriter:
    """
    Safe TAR writer.
    """

    @staticmethod
    def check(path: Union[str, Path], overwrite: bool = False):
        """
        Checks that a file has of the correct extension and and verifies that we can overwrite it if it exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if ".tar" not in path.suffixes:
            raise NameError(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)

    @staticmethod
    def write(
        dictionary: Dict[str, Any],
        path: Union[str, Path],
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        """
        Write a TAR archive of the dictionary, each key/value represents a single path name and associated file.

        :param dictionary: Dictionary to be archived.
        :param path: Path to the file.
        :param overwrite: Whether to overwrite, in case of an existing file.
        :param verbose: Verbosity of the method.
        :return:
        """
        path = Path(path)
        TarWriter.check(path=path, overwrite=overwrite)

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and overwrite:
            path.unlink()

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
