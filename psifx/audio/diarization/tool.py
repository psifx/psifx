"""speaker diarization tool."""

from typing import Union

from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

from pyannote.core.annotation import Segment, Annotation
from pyannote.core import notebook

from psifx.tool import Tool
from psifx.io import rttm


class DiarizationTool(Tool):
    """
    Base class for diarization tools.
    """

    def inference(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
    ):
        """
        Template of the inference method.

        :param audio_path: Path to the audio track.
        :param diarization_path: Path to the diarization file.
        :return:
        """
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
        """
        Plots a time series representing the diarized segments of an audio track.

        :param diarization_path: Path to the diarization file.
        :param visualization_path: Path to the visualization file.
        :return:
        """
        diarization_path = Path(diarization_path)
        visualization_path = Path(visualization_path)

        if self.verbose:
            print(f"diarization     =   {diarization_path}")
            print(f"visualization   =   {visualization_path}")

        rttm.RTTMReader.check(path=diarization_path)

        segments = rttm.RTTMReader.read(path=diarization_path, verbose=True)

        annotation = Annotation.from_records(
            iter(
                [
                    (
                        Segment(
                            start=segment["start"],
                            end=segment["start"] + segment["duration"],
                        ),
                        index,
                        segment["speaker_name"],
                    )
                    for index, segment in enumerate(
                        tqdm(
                            segments,
                            desc="Parsing",
                            disable=not self.verbose,
                        )
                    )
                ]
            )
        )

        if visualization_path.exists():
            if self.overwrite:
                visualization_path.unlink()
            else:
                raise FileExistsError(visualization_path)
        visualization_path.parent.mkdir(parents=True, exist_ok=True)

        plt.rcParams["figure.figsize"] = (notebook.width, 2)
        _, ax = plt.subplots()
        notebook.plot_annotation(annotation, ax=ax)
        plt.savefig(visualization_path, bbox_inches="tight")
