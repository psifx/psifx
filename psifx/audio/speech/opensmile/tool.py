"""
openSMILE speech processing tool.
"""

from typing import Union

from pathlib import Path

import numpy as np
import pandas as pd

from pydub import AudioSegment
import opensmile

from psifx.audio.speech.tool import SpeechTool
from psifx.io import rttm, tar, wav


FEATURE_SETS = {
    "ComParE_2016": opensmile.FeatureSet.ComParE_2016,
    "GeMAPSv01a": opensmile.FeatureSet.GeMAPSv01a,
    "GeMAPSv01b": opensmile.FeatureSet.GeMAPSv01b,
    "eGeMAPSv01a": opensmile.FeatureSet.eGeMAPSv01a,
    "eGeMAPSv01b": opensmile.FeatureSet.eGeMAPSv01b,
    "eGeMAPSv02": opensmile.FeatureSet.eGeMAPSv02,
    "emobase": opensmile.FeatureSet.emobase,
}

FEATURE_LEVELS = {
    "lld": opensmile.FeatureLevel.LowLevelDescriptors,
    "lld_de": opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
    "func": opensmile.FeatureLevel.Functionals,
}


def audio_segment_to_waveform(audio: AudioSegment) -> np.ndarray:
    """
    Converts an audio segment to a ndarray normalized waveform.

    :param audio: Audio segment
    :return: ndarray waveform.
    """
    samples = [c.get_array_of_samples() for c in audio.split_to_mono()]
    waveform = np.array(samples, dtype=np.float32) / np.iinfo(samples[0].typecode).max
    return waveform


class OpenSmileSpeechTool(SpeechTool):
    """
    openSMILE speech processing tool.
    """

    def __init__(
        self,
        feature_set: str = "ComParE_2016",
        feature_level: str = "func",
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device="cpu",
            overwrite=overwrite,
            verbose=verbose,
        )

        self.smile = opensmile.Smile(
            feature_set=FEATURE_SETS[feature_set],
            feature_level=FEATURE_LEVELS[feature_level],
        )

    def inference(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        features_path: Union[str, Path],
    ):
        """
        openSMILE inference method.

        :param audio_path: Path to the audio track.
        :param diarization_path: Path to the diarization file.
        :param features_path: Path to the feature file.
        :return:
        """
        audio_path = Path(audio_path)
        diarization_path = Path(diarization_path)
        features_path = Path(features_path)

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"diarization     =   {diarization_path}")
            print(f"features        =   {features_path}")

        wav.WAVReader.check(path=audio_path)
        rttm.RTTMReader.check(path=diarization_path)
        tar.TarWriter.check(path=features_path, overwrite=self.overwrite)

        audio = AudioSegment.from_wav(audio_path)

        segments = rttm.RTTMReader.read(diarization_path)

        features = []
        for segment in segments:
            start = segment["start"]
            end = segment["start"] + segment["duration"]
            start = int(start * 1000)
            end = int(end * 1000)
            cropped = audio[start:end].fade_in(duration=30).fade_out(duration=30)
            waveform = audio_segment_to_waveform(audio=cropped)
            feature = self.smile.process_signal(
                signal=waveform,
                sampling_rate=audio.frame_rate,
            )
            features.append(feature)
        features = pd.concat(features)

        tar.TarWriter.write(
            dictionary={"features.csv": features.to_csv()},
            path=features_path,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )
