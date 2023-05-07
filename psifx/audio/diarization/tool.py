from typing import Union

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from pyannote.core.annotation import Segment, Annotation
from pyannote.core import notebook

from psifx.tool import BaseTool
from psifx.io import rttm


class DiarizationTool(BaseTool):
    def inference(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
    ):
        audio_path = Path(audio_path)
        diarization_path = Path(diarization_path)

        # audio = load(audio_path)
        # audio = pre_process_func(audio)
        # diarization = model(audio)
        # diarization = post_process_func(diarization)
        # write(diarization, diarization_path)

        raise NotImplementedError

    def visualization(
        self,
        diarization_path: Union[str, Path],
        visualization_path: Union[str, Path],
    ):
        diarization_path = Path(diarization_path)
        visualization_path = Path(visualization_path)

        if self.verbose:
            print(f"diarization     =   {diarization_path}")
            print(f"visualization   =   {visualization_path}")

        dataframe = rttm.RTTMReader.read(path=diarization_path)
        dataframe["end"] = dataframe["start"] + dataframe["duration"]

        annotation = Annotation.from_records(
            iter(
                [
                    (
                        Segment(start=row["start"], end=row["end"]),
                        index,
                        row["speaker_name"],
                    )
                    for index, row in enumerate(dataframe.iloc)
                ]
            )
        )
        plt.rcParams["figure.figsize"] = (notebook.width, 2)
        fig, ax = plt.subplots()
        notebook.plot_annotation(annotation, ax=ax)
        plt.savefig(visualization_path, bbox_inches="tight")


def visualization_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diarization",
        type=Path,
        required=True,
        help="Path to the output diarization or directory containing the diarizations.",
    )
    parser.add_argument(
        "--visualization",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing files, otherwise raises an error.",
    )
    parser.add_argument(
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Verbosity of the script.",
    )
    args = parser.parse_args()

    tool = DiarizationTool(
        device="cpu",
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.visualization(
        diarization_path=args.diarization,
        visualization_path=args.visualization,
    )
    del tool
