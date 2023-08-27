from typing import Union

from pathlib import Path


class WAVReader:
    """
    Safe WAV reader.
    """

    @staticmethod
    def check(path: Union[str, Path]):
        """
        Checks that a file exists and is of the correct format.
        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".wav":
            raise NameError(path)
        if not path.exists():
            raise FileNotFoundError(path)


class WAVWriter:
    """
    Safe WAV writer.
    """

    @staticmethod
    def check(path: Union[str, Path]):
        """
        Checks that a file is of the correct format.
        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".wav":
            raise NameError(path)
