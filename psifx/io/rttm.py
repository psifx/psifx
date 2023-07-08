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


class RTTMReader(object):
    @staticmethod
    def check(path: Union[str, Path]):
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
        path = Path(path)
        RTTMReader.check(path)

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


class RTTMWriter(object):
    @staticmethod
    def check(path: Union[str, Path]):
        path = Path(path)
        if path.suffix != ".rttm":
            raise NameError(path)

    @staticmethod
    def write(
        path: Union[str, Path],
        segments: List[Dict[str, Union[str, float]]],
        overwrite: bool = False,
    ):
        path = Path(path)
        RTTMWriter.check(path)

        dataframe = pd.DataFrame.from_records(segments)

        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(
            path,
            sep=" ",
            na_rep="<NA>",
            columns=COLUMN_NAMES,
            header=False,
            index=False,
            float_format="%.3f",
        )
