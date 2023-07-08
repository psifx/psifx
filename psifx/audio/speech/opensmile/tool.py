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
    samples = [c.get_array_of_samples() for c in audio.split_to_mono()]
    waveform = np.array(samples, dtype=np.float32) / np.iinfo(samples[0].typecode).max
    return waveform


class OpenSmileSpeechTool(SpeechTool):
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
        audio_path = Path(audio_path)
        diarization_path = Path(diarization_path)
        features_path = Path(features_path)

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"diarization     =   {diarization_path}")
            print(f"features        =   {features_path}")

        wav.WAVReader.check(audio_path)
        rttm.RTTMReader.check(diarization_path)
        tar.TarWriter.check(features_path)

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
            feature = self.smile.process_signal(signal=waveform, sampling_rate=audio.frame_rate,)
            features.append(feature)
        features = pd.concat(features)

        tar.TarWriter.write(
            dictionary={"features.csv": features.to_csv()},
            path=features_path,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )


def inference_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to the audio file.",
    )
    parser.add_argument(
        "--diarization",
        type=Path,
        required=True,
        help="Path to the diarization file.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to the features file.",
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="ComParE_2016",
        help=f"Available sets: {list(FEATURE_SETS.keys())}",
    )
    parser.add_argument(
        "--feature_level",
        type=str,
        default="func",
        help=f"Available levels: {list(FEATURE_LEVELS.keys())}",
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

    tool = OpenSmileSpeechTool(
        feature_set=args.feature_set,
        feature_level=args.feature_level,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.inference(
        audio_path=args.audio,
        diarization_path=args.diarization,
        features_path=args.features,
    )
    del tool

