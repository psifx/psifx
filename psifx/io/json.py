"""JSON I/O module."""

from typing import Dict, List, Union

import json
from pathlib import Path
from tqdm import tqdm


class JSONReader:
    """
    Safe JSON reader.
    """

    @staticmethod
    def check(path: Union[str, Path]):
        """
        Checks that a file has of the correct extension and exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".json":
            raise NameError(path)
        if not path.exists():
            raise FileNotFoundError(path)

    @staticmethod
    def read(
        path: Union[str, Path],
        verbose: Union[bool, int] = True,
    ) -> Union[List, Dict]:
        """
        Reads and parses a JSON file.

        :param path: Path to the file.
        :param verbose: Verbosity of the method.
        :return: Deserialized data.
        """
        path = Path(path)
        JSONReader.check(path=path)

        with path.open(mode="r") as file:
            for _ in tqdm(
                range(1),
                desc="Reading",
                disable=not verbose,
            ):
                data = json.load(file)

        return data


class JSONWriter:
    """
    Safe JSON writer.
    """

    @staticmethod
    def check(path: Union[str, Path], overwrite: bool = False):
        """
        Checks that a file has of the correct extension and and verifies that we can overwrite it if it exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".json":
            raise NameError(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)

    @staticmethod
    def write(
        data: Union[List, Dict],
        path: Union[str, Path],
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        """
        Writes serializable data into a JSON file.

        :param data: Serializable data.
        :param path: Path to the file.
        :param overwrite: Whether to overwrite, in case of an existing file.
        :param verbose: Verbosity of the method.
        :return:
        """
        path = Path(path)
        JSONWriter.check(path=path, overwrite=overwrite)

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and overwrite:
            path.unlink()

        with path.open(mode="w") as file:
            for _ in tqdm(range(1), desc="Writing", disable=not verbose):
                json.dump(data, file)
