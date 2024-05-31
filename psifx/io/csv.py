from typing import Union
from pathlib import Path
import pandas as pd


class CsvReader:
    """
    Safe CSV file reader.
    """

    @staticmethod
    def check(path: Union[str, Path]):
        """
        Checks that a file has the correct extension and exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".csv":
            raise NameError(path)
        if not path.exists():
            raise FileNotFoundError(path)

    @staticmethod
    def read(
        path: Union[str, Path],
        verbose: Union[bool, int] = True,
    ) -> pd.DataFrame:
        """
        Reads and parses a CSV file.
        :param path: Path to the file.
        :param verbose: Verbosity of the method.
        :return: DataFrame with CSV content.
        """
        path = Path(path)
        CsvReader.check(path)

        df = pd.read_csv(path)

        return df


class CsvWriter:
    """
    Safe CSV file writer.
    """

    @staticmethod
    def check(path: Union[str, Path], overwrite: bool = False):
        """
        Checks that a file has the correct extension and verifies that we can overwrite it if it exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".csv":
            raise NameError(f"File {path} is not a CSV file.")
        if path.exists() and not overwrite:
            raise FileExistsError(path)

    @staticmethod
    def write(
        df: pd.DataFrame,
        path: Union[str, Path],
        overwrite: bool = False,
    ):
        """
        Writes DataFrame content into a CSV file.

        :param df: DataFrame with content.
        :param path: Path to the file.
        :param overwrite: Whether to overwrite, in case of an existing file.
        :return:
        """
        path = Path(path)
        CsvWriter.check(path, overwrite)

        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(f"File {path} already exists.")
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index=False)
