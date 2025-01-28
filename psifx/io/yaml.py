"""YAML I/O module."""

from typing import Dict, List, Union

import yaml
from pathlib import Path
from tqdm import tqdm


class YAMLReader:
    """
    Safe YAML reader.
    """

    @staticmethod
    def check(path: Union[str, Path]):
        """
        Checks that a file has the correct extension and exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix not in [".yaml", ".yml"]:
            raise NameError(f"Path {path} does not have a .csv extension.")
        if not path.exists():
            raise FileNotFoundError(f"File missing at path {path}")

    @staticmethod
    def read(
        path: Union[str, Path],
        verbose: Union[bool, int] = True,
    ) -> Union[List, Dict]:
        """
        Reads and parses a YAML file.

        :param path: Path to the file.
        :param verbose: Verbosity of the method.
        :return: Deserialized data.
        """
        path = Path(path)
        YAMLReader.check(path=path)

        with path.open(mode="r") as file:
            for _ in tqdm(
                range(1),
                desc="Reading",
                disable=not verbose,
            ):
                data = yaml.safe_load(file)

        return data


class YAMLWriter:
    """
    Safe YAML writer.
    """

    @staticmethod
    def check(path: Union[str, Path], overwrite: bool = False):
        """
        Checks that a file has the correct extension and verifies that we can overwrite it if it exists.

        :param path: Path to the file.
        :param overwrite: Whether to overwrite, in case of an existing file.
        :return:
        """
        path = Path(path)
        if path.suffix not in [".yaml", ".yml"]:
            raise NameError(f"Path {path} does not have a .yaml extension.")
        if path.exists() and not overwrite:
            raise FileExistsError(f"File {path} already exists.")

    @staticmethod
    def write(
        data: Union[List, Dict],
        path: Union[str, Path],
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        """
        Writes serializable data into a YAML file.

        :param data: Serializable data.
        :param path: Path to the file.
        :param overwrite: Whether to overwrite, in case of an existing file.
        :param verbose: Verbosity of the method.
        :return:
        """
        path = Path(path)
        YAMLWriter.check(path=path, overwrite=overwrite)

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and overwrite:
            path.unlink()

        with path.open(mode="w") as file:
            for _ in tqdm(range(1), desc="Writing", disable=not verbose):
                yaml.safe_dump(data, file)
