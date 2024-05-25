"""RTTM I/O module."""

from typing import Dict, List, Union

from pathlib import Path
from tqdm import tqdm

import pandas as pd


COLUMN_NAMES = [
    "type",
    "file_stem",
    "channel",
    "start",
    "duration",
    "orthography",
    "speaker_type",
    "speaker_name",
    "confidence_score",
    "signal_lookahead_time",
]


class RTTMReader:
    """
    Safe RTTM reader.
    """

    @staticmethod
    def check(path: Union[str, Path]):
        """
        Checks that a file has of the correct extension and exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".rttm":
            raise NameError(path)
        if not path.exists():
            raise FileNotFoundError(path)

    @staticmethod
    def read(
        path: Union[str, Path],
        verbose: bool = True,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Reads and parse an RTTM file.

        :param path: Path to the file.
        :param verbose: Verbosity of the method.
        :return:
        """
        path = Path(path)
        RTTMReader.check(path=path)

        dataframe = pd.read_csv(
            path,
            sep=" ",
            header=None,
            names=COLUMN_NAMES,
        )
        segments = [
            {
                "type": dataframe.iloc[index]["type"],
                "file_stem": dataframe.iloc[index]["file_stem"],
                "channel": dataframe.iloc[index]["channel"],
                "start": dataframe.iloc[index]["start"],
                "duration": dataframe.iloc[index]["duration"],
                "orthography": dataframe.iloc[index]["orthography"],
                "speaker_type": dataframe.iloc[index]["speaker_type"],
                "speaker_name": dataframe.iloc[index]["speaker_name"],
                "confidence_score": dataframe.iloc[index]["confidence_score"],
                "signal_lookahead_time": dataframe.iloc[index]["signal_lookahead_time"],
            }
            for index in tqdm(
                range(dataframe.shape[0]),
                desc="Decoding",
                disable=not verbose,
            )
        ]
        return segments


class RTTMWriter:
    """
    Safe RTTM writer.
    """

    @staticmethod
    def check(path: Union[str, Path], overwrite: bool = False):
        """
        Checks that a file has of the correct extension and and verifies that we can overwrite it if it exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".rttm":
            raise NameError(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)

    @staticmethod
    def write(
        segments: List[Dict[str, Union[str, float]]],
        path: Union[str, Path],
        overwrite: bool = False,
    ):
        """
        Writes diarized audio segment information in the RTTM format.

        :param segments: Audio segment information.
        :param path: Path to the file.
        :param overwrite: Whether to overwrite, in case of an existing file.
        :return:
        """
        path = Path(path)
        RTTMWriter.check(path=path, overwrite=overwrite)

        dataframe = pd.DataFrame.from_records(segments)

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and overwrite:
            path.unlink()

        dataframe.to_csv(
            path,
            sep=" ",
            na_rep="<NA>",
            columns=COLUMN_NAMES,
            header=False,
            index=False,
            float_format="%.3f",
        )
