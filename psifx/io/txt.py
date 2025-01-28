from typing import Union
from pathlib import Path


class TxtReader:
    """
    Safe text file reader.
    """

    @staticmethod
    def check(path: Union[str, Path]):
        """
        Checks that a file has the correct extension and exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".txt":
            raise NameError(f"Path {path} does not have a .txt extension.")
        if not path.exists():
            raise FileNotFoundError(f"File missing at path {path}")

    @staticmethod
    def read(
            path: Union[str, Path],
            verbose: Union[bool, int] = True,
    ) -> str:
        """
        Reads and parses a text file.
        :param path: Path to the file.
        :param verbose: Verbosity of the method.
        :return: Text content.
        """
        path = Path(path)
        TxtReader.check(path)

        with path.open(mode="r") as file:
            content = file.read()

        return content


class TxtWriter:
    """
    Safe text file writer.
    """

    @staticmethod
    def check(path: Union[str, Path], overwrite: bool = False):
        """
        Checks that a file has the correct extension.
        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".txt":
            raise NameError(f"Path {path} does not have a .txt extension.")
        if path.exists() and not overwrite:
            raise FileExistsError(f"File {path} already exists.")

    @staticmethod
    def write(
            content: str,
            path: Union[str, Path],
            overwrite: bool = False,
    ):
        """
        Writes text content into a text file.

        :param content: Text content.
        :param path: Path to the file.
        :param overwrite: Whether to overwrite, in case of an existing file.
        :return:
        """
        path = Path(path)
        TxtWriter.check(path, overwrite)

        if path.exists():
            if overwrite:
                path.unlink()

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode="w") as file:
            file.write(content)
