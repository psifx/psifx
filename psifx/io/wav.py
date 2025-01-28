"""WAV I/O module."""

from typing import Union

from pathlib import Path


class WAVReader:
    """
    Safe WAV reader.
    """

    @staticmethod
    def check(path: Union[str, Path]):
        """
        Checks that a file has the correct extension and exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".wav":
            raise NameError(f"Path {path} does not have a .wav extension.")
        if not path.exists():
            raise FileNotFoundError(f"File missing at path {path}")


class WAVWriter:
    """
    Safe WAV writer.
    """

    @staticmethod
    def check(path: Union[str, Path], overwrite: bool = False):
        """
        Checks that a file has of the correct extension and verifies that we can overwrite it if it exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".wav":
            raise NameError(f"Path {path} does not have a .wav extension.")
        if path.exists() and not overwrite:
            raise FileExistsError(f"File {path} already exists.")
